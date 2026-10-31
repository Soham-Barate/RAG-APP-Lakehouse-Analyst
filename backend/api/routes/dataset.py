"""
FastAPI endpoints for user dataset lifecycle management:
Uploads, automated profiling, schema inspection, and dataset isolation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)

from backend.api.middleware.auth import verify_api_key
from backend.dataset.indexer import DatasetTextIndexer
from backend.dataset.jobs import JobStage, get_job_manager
from backend.dataset.manager import (
    DEMO_DATASET_ID,
    DatasetValidationError,
    get_dataset_manager,
)
from backend.dataset.models import DataQualityReport, DatasetMetadata
from backend.dataset.streaming import (
    StreamingIngestionError,
    get_upload_session_manager,
    stream_upload_file,
)
from pydantic import BaseModel, Field


class UploadInitRequest(BaseModel):
    filename: str
    total_bytes: int = Field(ge=1, le=5 * 1024 * 1024 * 1024)
    chunk_size: int = Field(default=2 * 1024 * 1024, ge=1, le=16 * 1024 * 1024)
    checksum_sha256: str | None = None


class UploadFinalizeRequest(BaseModel):
    upload_id: str

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/datasets", tags=["Datasets"])


@router.get(
    "",
    response_model=list[DatasetMetadata],
    summary="List all available datasets (demo + user uploaded)",
)
async def list_datasets() -> list[DatasetMetadata]:
    """Retrieve all accessible datasets with row/column counts and table identifiers."""
    manager = get_dataset_manager()
    return manager.list_datasets()


@router.post(
    "/upload/init",
    summary="Initialize a multi-part chunked streaming upload session",
)
async def init_chunked_upload(
    payload: UploadInitRequest,
    api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Start a new chunked streaming upload session with size and format guardrails."""
    ext = Path(payload.filename).suffix.lower()
    from backend.dataset.streaming import ALLOWED_EXTENSIONS
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    if ext in (".xlsx", ".xls") and payload.total_bytes > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Excel (.xlsx/.xls) files are limited to 50 MB due to XML decompression limits. Please export to CSV or Parquet.",
        )
    session_mgr = get_upload_session_manager()
    session = session_mgr.init_session(
        filename=payload.filename,
        total_bytes=payload.total_bytes,
        chunk_size=payload.chunk_size,
        expected_checksum=payload.checksum_sha256,
    )
    return {
        "upload_id": session.upload_id,
        "filename": session.filename,
        "total_bytes": session.total_bytes,
        "chunk_size": session.chunk_size,
        "total_chunks": session.total_chunks,
        "status": "initialized",
    }


@router.post(
    "/upload/chunk",
    summary="Upload a single binary chunk in a chunked upload session",
)
async def upload_file_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Append a chunk directly to disk without holding previous chunks in RAM."""
    session_mgr = get_upload_session_manager()
    try:
        chunk_bytes = await file.read()
        res = session_mgr.append_chunk(upload_id, chunk_index, chunk_bytes)
        return res
    except StreamingIngestionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/upload/finalize",
    summary="Finalize chunked upload, verify checksum, and ingest to Silver Parquet",
)
async def finalize_chunked_upload(
    request: Request,
    payload: UploadFinalizeRequest,
    api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Verify complete upload assembly and trigger streaming Parquet conversion and profiling."""
    session_mgr = get_upload_session_manager()
    try:
        session = session_mgr.get_session(payload.upload_id)
    except StreamingIngestionError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    manager = get_dataset_manager()
    temp_target = Path("data/uploads/temp") / f"final_{session.upload_id}_{session.filename}"

    try:
        final_file, sha256_hash = session_mgr.finalize_session(payload.upload_id, temp_target)
        metadata = manager.ingest_from_disk_streaming(
            raw_file_path=final_file,
            original_filename=session.filename,
        )
    except Exception as exc:
        if temp_target.exists():
            try:
                temp_target.unlink()
            except OSError:
                pass
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ingestion conversion failed: {exc}",
        ) from exc
    finally:
        if temp_target.exists():
            try:
                temp_target.unlink()
            except OSError:
                pass

    profile = manager.get_profiling_report(metadata.dataset_id)
    return {
        "success": True,
        "metadata": metadata.model_dump(),
        "profile": profile.model_dump() if profile else None,
        "sha256": sha256_hash,
    }


@router.post(
    "/upload",
    response_model=dict[str, Any],
    summary="Upload and profile a new dataset (CSV, XLSX, Parquet, JSON)",
)
async def upload_dataset(
    request: Request,
    file: UploadFile,
    api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """
    Upload a user dataset. Performs extension and magic-byte validation,
    sanitizes column identifiers, persists Parquet representation,
    generates automated data profiling, and registers an isolated DuckDB table.
    Streams directly to disk with bounded memory (<4 MB buffer) to support up to 5 GB.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename must be provided.",
        )

    upload_temp_dir = Path("data/uploads/.tmp")
    upload_temp_dir.mkdir(parents=True, exist_ok=True)
    temp_upload_file = upload_temp_dir / f"upload_{uuid.uuid4().hex[:8]}_{file.filename}"

    async def file_chunk_generator():
        while True:
            chunk = await file.read(4 * 1024 * 1024)  # 4 MB chunk
            if not chunk:
                break
            yield chunk

    manager = get_dataset_manager()
    try:
        await stream_upload_file(
            file_chunks=file_chunk_generator(),
            destination_path=temp_upload_file,
        )
        metadata = manager.ingest_from_disk_streaming(
            raw_file_path=temp_upload_file,
            original_filename=file.filename,
        )
    except DatasetValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during streaming upload: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal dataset ingestion error: {exc}",
        ) from exc
    finally:
        if temp_upload_file.exists():
            try:
                temp_upload_file.unlink()
            except OSError:
                pass

    # Retrieve profiling report
    profile = manager.get_profiling_report(metadata.dataset_id)

    # If text columns exist, optionally trigger text indexing
    indexed_count = 0
    vector_store = getattr(request.app.state, "vector_store", None)
    embedding_service = getattr(request.app.state, "embedding_service", None)
    if vector_store and embedding_service and metadata.has_text_indexed:
        try:
            indexer = DatasetTextIndexer(
                vector_store=vector_store,
                embedding_service=embedding_service,
                dataset_manager=manager,
            )
            indexed_count = await indexer.index_dataset_text(metadata.dataset_id, max_rows=200)
        except Exception as e:
            logger.warning("Optional vector text indexing failed for %s: %s", metadata.dataset_id, e)

    return {
        "success": True,
        "metadata": metadata.model_dump(),
        "profile": profile.model_dump() if profile else None,
        "text_chunks_indexed": indexed_count,
    }


@router.get(
    "/{dataset_id}",
    summary="Get detailed metadata, profiling report, and preview for a dataset",
)
async def get_dataset_detail(
    dataset_id: str,
) -> dict[str, Any]:
    """Fetch complete statistics, quality warnings, column types, and sample rows."""
    manager = get_dataset_manager()
    meta = manager.get_dataset(dataset_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found.",
        )

    profile = manager.get_profiling_report(dataset_id)
    preview = manager.get_sample_rows(dataset_id, limit=50)

    return {
        "metadata": meta.model_dump(),
        "profile": profile.model_dump() if profile else None,
        "preview": preview,
    }


@router.get(
    "/{dataset_id}/preview",
    summary="Get sample records from dataset",
)
async def get_dataset_preview(
    dataset_id: str,
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Retrieve up to 500 rows for data inspection."""
    manager = get_dataset_manager()
    meta = manager.get_dataset(dataset_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found.",
        )
    return manager.get_sample_rows(dataset_id, limit=limit)


@router.post(
    "/regenerate-demo",
    summary="Regenerate synthetic e-commerce lakehouse demo data with fresh transactions",
)
async def regenerate_demo_dataset(
    scale: str = Query(default="quick", pattern="^(quick|medium|large|enterprise)$"),
) -> dict[str, Any]:
    """
    Regenerate the synthetic demo dataset with fresh random seed at the chosen scale.
    Refreshes orders, revenue, customer spend, and re-attaches DuckDB views.
    Scales:
      - quick: ~50K rows, ~5 MB Parquet
      - medium: ~2M rows, ~200 MB Parquet
      - large: ~10M rows, ~1 GB Parquet
      - enterprise: ~50M rows, ~5 GB Parquet
    """
    try:
        from ingestion.synthetic.data_generator import SyntheticDataGenerator
        from backend.rag.nodes.sql_node import SafeSQLNode

        t0 = time.perf_counter()
        SyntheticDataGenerator.regenerate_new_data(scale=scale)
        SafeSQLNode.reset_connection()
        duration = round(time.perf_counter() - t0, 2)

        orders_path = Path("data/silver/orders.parquet")
        orders_size_mb = round(orders_path.stat().st_size / (1024 * 1024), 2) if orders_path.exists() else 0.0

        return {
            "success": True,
            "scale": scale,
            "duration_seconds": duration,
            "orders_parquet_size_mb": orders_size_mb,
            "message": f"Demo lakehouse data regenerated at scale '{scale}' in {duration}s ({orders_size_mb} MB).",
        }
    except Exception as exc:
        logger.error("Failed to regenerate synthetic data: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data regeneration failed: {exc}",
        ) from exc


@router.post(
    "/regenerate-async",
    summary="Asynchronously regenerate synthetic lakehouse data in background with live progress tracking",
)
async def regenerate_demo_async(
    background_tasks: BackgroundTasks,
    scale: str = Query(default="quick", pattern="^(quick|medium|large|enterprise)$"),
) -> dict[str, Any]:
    """Start asynchronous dataset generation in background with durable progress tracking."""
    job_mgr = get_job_manager()
    job = job_mgr.create_job(
        job_type="synthetic_generation",
        dataset_id=DEMO_DATASET_ID,
        details={"scale": scale},
    )

    def _run_bg_generation(job_id: str, target_scale: str):
        from ingestion.synthetic.data_generator import SyntheticDataGenerator
        from backend.rag.nodes.sql_node import SafeSQLNode

        def progress_cb(msg: str, pct: float):
            job_mgr.update_stage(
                job_id=job_id,
                stage=JobStage.SILVER_PROCESSING if pct < 0.9 else JobStage.GOLD_PROCESSING,
                progress_pct=round(pct * 100, 1),
                message=msg,
            )

        try:
            SyntheticDataGenerator.regenerate_new_data(scale=target_scale, progress_cb=progress_cb)
            SafeSQLNode.reset_connection()
            orders_path = Path("data/silver/orders.parquet")
            size_mb = round(orders_path.stat().st_size / (1024 * 1024), 2) if orders_path.exists() else 0.0
            job_mgr.complete_job(job_id, {"orders_parquet_size_mb": size_mb, "scale": target_scale})
        except Exception as exc:
            job_mgr.fail_job(job_id, str(exc))

    background_tasks.add_task(_run_bg_generation, job.job_id, scale)
    return {
        "success": True,
        "job_id": job.job_id,
        "scale": scale,
        "status": "running",
        "message": f"Background generation started at scale '{scale}'. Track progress via /api/v1/datasets/jobs/{job.job_id}",
    }


@router.get(
    "/jobs/{job_id}",
    summary="Check status and real-time progress percentage of an asynchronous ingestion/generation job",
)
async def get_job_status(job_id: str) -> dict[str, Any]:
    """Poll job progress percentage and stage."""
    job_mgr = get_job_manager()
    job = job_mgr.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return job.to_dict()


@router.get(
    "/jobs",
    summary="List recent background ingestion and generation jobs",
)
async def list_jobs(limit: int = Query(default=20, ge=1, le=100)) -> list[dict[str, Any]]:
    """List recent background jobs."""
    job_mgr = get_job_manager()
    return [j.to_dict() for j in job_mgr.list_jobs(limit=limit)]
    """
    Regenerate the synthetic demo dataset with fresh random seed.
    Refreshes orders, revenue, customer spend, and re-attaches DuckDB views.
    """
    try:
        from ingestion.synthetic.data_generator import SyntheticDataGenerator
        from backend.rag.nodes.sql_node import SafeSQLNode

        SyntheticDataGenerator.regenerate_new_data()
        SafeSQLNode.reset_connection()

        return {
            "success": True,
            "message": "Demo lakehouse data regenerated with fresh transactions and updated revenue.",
        }
    except Exception as exc:
        logger.error("Failed to regenerate synthetic data: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data regeneration failed: {exc}",
        ) from exc


@router.delete(
    "/{dataset_id}",
    summary="Delete a user-uploaded dataset and purge disk artifacts",
)
async def delete_dataset(
    dataset_id: str,
    api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Safely drop user dataset files, manifest entry, and DuckDB registration."""
    if dataset_id == DEMO_DATASET_ID:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete the built-in demo lakehouse dataset.",
        )

    manager = get_dataset_manager()
    success = manager.delete_dataset(dataset_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found.",
        )

    return {"success": True, "message": f"Dataset '{dataset_id}' deleted successfully."}
