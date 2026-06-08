"""
Data ingestion API endpoints.
"""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Request

from backend.api.middleware.auth import verify_api_key
from backend.core.schemas import IngestRequest, IngestResponse

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])


async def _run_ingestion(
    request_data: dict,
    app_state: object,
) -> None:
    """Background task for data ingestion."""
    from ingestion.pipeline import IngestionPipeline

    try:
        pipeline = IngestionPipeline(app_state=app_state)
        await pipeline.run(
            source_type=request_data["source_type"],
            source_path=request_data.get("source_path"),
            table_name=request_data.get("table_name"),
            document_type=request_data.get("document_type"),
            batch_size=request_data.get("batch_size", 100),
        )
        await logger.ainfo(
            "ingestion_completed",
            job_id=request_data["job_id"],
        )
    except Exception as exc:
        await logger.aerror(
            "ingestion_failed",
            job_id=request_data["job_id"],
            error=str(exc),
        )


@router.post("", response_model=IngestResponse)
async def start_ingestion(
    request: Request,
    body: IngestRequest,
    background_tasks: BackgroundTasks,
    _api_key: str = Depends(verify_api_key),
) -> IngestResponse:
    """
    Start a data ingestion job.

    Runs in the background so the API returns immediately. The job
    processes documents, chunks them, generates embeddings, and upserts
    into the configured vector store(s).
    """
    job_id = str(uuid.uuid4())

    await logger.ainfo(
        "ingestion_started",
        job_id=job_id,
        source_type=body.source_type.value,
        source_path=body.source_path,
        table_name=body.table_name,
    )

    request_data = {
        "job_id": job_id,
        "source_type": body.source_type.value,
        "source_path": body.source_path,
        "table_name": body.table_name,
        "document_type": body.document_type.value if body.document_type else None,
        "batch_size": body.batch_size,
    }

    background_tasks.add_task(
        _run_ingestion,
        request_data=request_data,
        app_state=request.app.state,
    )

    return IngestResponse(
        job_id=job_id,
        status="started",
    )


@router.post("/synthetic", response_model=IngestResponse)
async def ingest_synthetic_data(
    request: Request,
    background_tasks: BackgroundTasks,
    _api_key: str = Depends(verify_api_key),
) -> IngestResponse:
    """
    Generate and ingest synthetic e-commerce demo data.

    Creates realistic products, orders, reviews, and support tickets,
    then ingests them into the vector store for immediate testing.
    """
    job_id = str(uuid.uuid4())

    await logger.ainfo("synthetic_ingestion_started", job_id=job_id)

    async def _generate_and_ingest() -> None:
        from ingestion.synthetic.data_generator import SyntheticDataGenerator
        from ingestion.pipeline import IngestionPipeline

        try:
            generator = SyntheticDataGenerator()
            data_path = generator.generate_all()

            pipeline = IngestionPipeline(app_state=request.app.state)
            await pipeline.run(
                source_type="file",
                source_path=str(data_path),
                batch_size=100,
            )
            await logger.ainfo("synthetic_ingestion_completed", job_id=job_id)
        except Exception as exc:
            await logger.aerror(
                "synthetic_ingestion_failed",
                job_id=job_id,
                error=str(exc),
            )

    background_tasks.add_task(_generate_and_ingest)

    return IngestResponse(job_id=job_id, status="started")
