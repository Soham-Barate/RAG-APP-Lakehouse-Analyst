"""
File upload API endpoint.

Accepts multipart file uploads, validates them securely, saves to
disk, and optionally triggers the ingestion pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from backend.api.middleware.rbac import require_permission
from backend.core.file_validator import FileValidationError, FileValidator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Upload"])

_validator = FileValidator()


@router.post(
    "/upload",
    summary="Upload a file for ingestion",
    dependencies=[Depends(require_permission("ingest:upload"))],
)
async def upload_file(
    file: UploadFile,
) -> dict[str, Any]:
    """
    Upload a file for ingestion into the RAG knowledge base.

    Accepts: PDF, DOCX, CSV, TXT, JSON, JSONL, Markdown, Parquet.
    Max size varies by type (see file validator).
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "INVALID_REQUEST", "message": "No filename provided."},
        )

    try:
        # Read file content
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "EMPTY_FILE", "message": "Uploaded file is empty."},
            )

        # Validate and save
        save_path = await _validator.save_upload(file.filename, content)

        logger.info(
            "File uploaded: %s (%d bytes) → %s",
            file.filename,
            len(content),
            save_path,
        )

        return {
            "status": "uploaded",
            "filename": save_path.name,
            "size_bytes": len(content),
            "message": (
                f"File '{file.filename}' uploaded successfully. "
                f"Use POST /api/v1/ingest with source_type='file' and "
                f"source_path='{save_path}' to ingest it."
            ),
        }

    except FileValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": exc.error_code, "message": str(exc)},
        )
    except Exception as exc:
        logger.error("File upload failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "UPLOAD_ERROR", "message": f"Upload failed: {exc}"},
        )
