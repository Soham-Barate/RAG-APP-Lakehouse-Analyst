"""
Durable asynchronous job tracking system for dataset ingestion and generation.

Persists multi-stage job progress to disk (JSON) to guarantee durability across
restarts without requiring heavy external dependencies.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JOBS_DIR = Path("data/jobs")


class JobStage(str, Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    VALIDATING = "validating"
    BRONZE_PROCESSING = "bronze_processing"
    SILVER_PROCESSING = "silver_processing"
    GOLD_PROCESSING = "gold_processing"
    PARQUET_GENERATION = "parquet_generation"
    PROFILING = "profiling"
    VECTOR_INDEXING = "vector_indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobRecord:
    job_id: str
    job_type: str  # "upload_ingestion" or "synthetic_generation"
    dataset_id: str
    stage: str = JobStage.PENDING.value
    status: str = "running"  # "running", "completed", "failed"
    progress_pct: float = 0.0
    message: str = "Job initialized"
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_sec: float = 0.0
    rows_processed: int = 0
    total_rows: int = 0
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobManager:
    """Manages persistent lifecycle of asynchronous background ingestion jobs."""

    def __init__(self, storage_dir: Path = JOBS_DIR) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _job_file(self, job_id: str) -> Path:
        return self.storage_dir / f"{job_id}.json"

    def create_job(
        self,
        job_type: str,
        dataset_id: str,
        details: dict[str, Any] | None = None,
    ) -> JobRecord:
        """Create and persist a new background job."""
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        record = JobRecord(
            job_id=job_id,
            job_type=job_type,
            dataset_id=dataset_id,
            details=details or {},
        )
        self.save_job(record)
        return record

    def save_job(self, record: JobRecord) -> None:
        """Atomically persist job to disk."""
        target = self._job_file(record.job_id)
        temp = target.with_suffix(".tmp")
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)
        temp.replace(target)

    def get_job(self, job_id: str) -> JobRecord | None:
        """Retrieve job record by ID."""
        target = self._job_file(job_id)
        if not target.exists():
            return None
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
                return JobRecord(**data)
        except Exception as exc:
            logger.error("Failed reading job file %s: %s", target, exc)
            return None

    def update_stage(
        self,
        job_id: str,
        stage: JobStage | str,
        progress_pct: float,
        message: str = "",
        rows_processed: int = 0,
        total_rows: int = 0,
    ) -> None:
        """Update job stage and progress percentage."""
        job = self.get_job(job_id)
        if not job:
            return
        stage_val = stage.value if isinstance(stage, JobStage) else str(stage)
        job.stage = stage_val
        job.progress_pct = max(0.0, min(100.0, progress_pct))
        if message:
            job.message = message
        if rows_processed:
            job.rows_processed = rows_processed
        if total_rows:
            job.total_rows = total_rows
        job.duration_sec = round(time.time() - job.start_time, 2)
        self.save_job(job)

    def complete_job(self, job_id: str, details: dict[str, Any] | None = None) -> None:
        """Mark job as successfully completed."""
        job = self.get_job(job_id)
        if not job:
            return
        job.stage = JobStage.COMPLETED.value
        job.status = "completed"
        job.progress_pct = 100.0
        job.message = "Completed successfully."
        job.end_time = time.time()
        job.duration_sec = round(job.end_time - job.start_time, 2)
        if details:
            job.details.update(details)
        self.save_job(job)
        logger.info("Job %s completed in %.2fs", job_id, job.duration_sec)

    def fail_job(self, job_id: str, error: str) -> None:
        """Mark job as failed with error details."""
        job = self.get_job(job_id)
        if not job:
            return
        job.stage = JobStage.FAILED.value
        job.status = "failed"
        job.error = str(error)
        job.message = f"Failed: {error}"
        job.end_time = time.time()
        job.duration_sec = round(job.end_time - job.start_time, 2)
        self.save_job(job)
        logger.error("Job %s failed after %.2fs: %s", job_id, job.duration_sec, error)

    def list_jobs(self, limit: int = 50) -> list[JobRecord]:
        """List recent jobs ordered by start time descending."""
        jobs: list[JobRecord] = []
        for file in self.storage_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    jobs.append(JobRecord(**json.load(f)))
            except Exception:
                continue
        jobs.sort(key=lambda j: j.start_time, reverse=True)
        return jobs[:limit]


# Singleton
_default_job_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    global _default_job_manager
    if _default_job_manager is None:
        _default_job_manager = JobManager()
    return _default_job_manager

