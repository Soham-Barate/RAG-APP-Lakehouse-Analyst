"""
FastAPI endpoints for user dataset lifecycle management:
Uploads, automated profiling, schema inspection, and dataset isolation.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, status

from backend.api.middleware.auth import verify_api_key
from backend.dataset.indexer import DatasetTextIndexer
from backend.dataset.manager import (
    DEMO_DATASET_ID,
    DatasetValidationError,
    get_dataset_manager,
)
from backend.dataset.models import DataQualityReport, DatasetMetadata

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
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename must be provided.",
        )

    try:
        file_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed reading uploaded file: {exc}",
        ) from exc

    manager = get_dataset_manager()
    try:
        metadata = manager.upload_dataset(
            file_bytes=file_bytes,
            filename=file.filename,
            content_type=file.content_type,
        )
    except DatasetValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during dataset upload: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal dataset ingestion error: {exc}",
        ) from exc

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
