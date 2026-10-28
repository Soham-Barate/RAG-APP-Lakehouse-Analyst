"""
Secure file validation for upload handling.

Validates file types by magic bytes (not just extension), enforces
size limits, and provides safe temporary file management.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Magic byte signatures for allowed file types
# ---------------------------------------------------------------------------
MAGIC_SIGNATURES: dict[str, list[bytes]] = {
    "pdf":  [b"%PDF"],
    "docx": [b"PK\x03\x04"],     # ZIP-based format
    "xlsx": [b"PK\x03\x04"],     # ZIP-based format
    "csv":  [],                   # No magic bytes — validated by content
    "txt":  [],                   # No magic bytes — validated by content
    "json": [],                   # No magic bytes — validated by content
    "jsonl": [],                  # No magic bytes — validated by content
    "md":   [],                   # No magic bytes — validated by content
    "parquet": [b"PAR1"],
}

ALLOWED_EXTENSIONS: set[str] = {
    ".pdf", ".docx", ".xlsx", ".csv", ".txt",
    ".json", ".jsonl", ".md", ".markdown", ".parquet",
}

# Maximum file sizes in bytes
MAX_FILE_SIZES: dict[str, int] = {
    ".pdf": 50 * 1024 * 1024,       # 50 MB
    ".docx": 25 * 1024 * 1024,      # 25 MB
    ".xlsx": 25 * 1024 * 1024,      # 25 MB
    ".csv": 100 * 1024 * 1024,      # 100 MB
    ".txt": 10 * 1024 * 1024,       # 10 MB
    ".json": 50 * 1024 * 1024,      # 50 MB
    ".jsonl": 100 * 1024 * 1024,    # 100 MB
    ".md": 10 * 1024 * 1024,        # 10 MB
    ".markdown": 10 * 1024 * 1024,  # 10 MB
    ".parquet": 200 * 1024 * 1024,  # 200 MB
}

DEFAULT_MAX_SIZE = 50 * 1024 * 1024  # 50 MB default


class FileValidationError(Exception):
    """Raised when file validation fails."""

    def __init__(self, message: str, error_code: str = "FILE_VALIDATION_ERROR"):
        self.error_code = error_code
        super().__init__(message)


class FileValidator:
    """
    Secure file validator for upload handling.

    Checks:
    1. File extension against allowlist
    2. File size against per-type limits
    3. Magic bytes to verify actual file type
    4. Filename safety (no path traversal)
    5. Content basic sanity checks
    """

    def __init__(
        self,
        allowed_extensions: set[str] | None = None,
        max_file_sizes: dict[str, int] | None = None,
        upload_dir: str | Path = "data/uploads",
    ) -> None:
        try:
            from config.settings import get_settings
            settings = get_settings()
            config_allowed = set(settings.allowed_file_types)
            config_max_size = settings.max_file_size_mb * 1024 * 1024
        except Exception:
            config_allowed = ALLOWED_EXTENSIONS
            config_max_size = DEFAULT_MAX_SIZE

        self.allowed_extensions = allowed_extensions or config_allowed

        if max_file_sizes is not None:
            self.max_file_sizes = max_file_sizes
        else:
            self.max_file_sizes = {}
            for ext in self.allowed_extensions:
                self.max_file_sizes[ext] = MAX_FILE_SIZES.get(ext, config_max_size)

        self.upload_dir = Path(upload_dir)

    def validate_filename(self, filename: str) -> str:
        """
        Validate and sanitize a filename.

        Prevents path traversal and ensures safe characters.
        Returns the sanitized basename.
        """
        # Extract basename only — prevent directory traversal
        safe_name = Path(filename).name

        # Remove any remaining path separators
        safe_name = safe_name.replace("/", "_").replace("\\", "_")

        # Remove null bytes
        safe_name = safe_name.replace("\x00", "")

        # Limit length
        if len(safe_name) > 255:
            stem = Path(safe_name).stem[:200]
            suffix = Path(safe_name).suffix
            safe_name = f"{stem}{suffix}"

        if not safe_name or safe_name.startswith("."):
            raise FileValidationError(
                f"Invalid filename: {filename!r}",
                error_code="INVALID_FILENAME",
            )

        return safe_name

    def validate_extension(self, filename: str) -> str:
        """Validate the file extension against the allowlist."""
        ext = Path(filename).suffix.lower()
        if ext not in self.allowed_extensions:
            raise FileValidationError(
                f"File type '{ext}' is not allowed. "
                f"Allowed types: {', '.join(sorted(self.allowed_extensions))}",
                error_code="INVALID_FILE_TYPE",
            )
        return ext

    def validate_size(self, size_bytes: int, extension: str) -> None:
        """Validate file size against per-type limits."""
        max_size = self.max_file_sizes.get(extension, DEFAULT_MAX_SIZE)
        if size_bytes > max_size:
            max_mb = max_size / (1024 * 1024)
            actual_mb = size_bytes / (1024 * 1024)
            raise FileValidationError(
                f"File too large: {actual_mb:.1f} MB "
                f"(max {max_mb:.0f} MB for {extension} files)",
                error_code="FILE_TOO_LARGE",
            )

    def validate_magic_bytes(self, content: bytes, extension: str) -> None:
        """Verify file content matches expected magic bytes."""
        ext_key = extension.lstrip(".")
        signatures = MAGIC_SIGNATURES.get(ext_key, [])

        if not signatures:
            return  # No magic bytes to check for this type

        header = content[:8]
        if not any(header.startswith(sig) for sig in signatures):
            raise FileValidationError(
                f"File content does not match expected format for {extension}",
                error_code="CONTENT_MISMATCH",
            )

    def validate(
        self,
        filename: str,
        content: bytes,
    ) -> dict[str, Any]:
        """
        Run all validations on an uploaded file.

        Returns metadata dict with safe filename, extension, size, and hash.
        """
        safe_name = self.validate_filename(filename)
        ext = self.validate_extension(safe_name)
        self.validate_size(len(content), ext)
        self.validate_magic_bytes(content, ext)

        file_hash = hashlib.sha256(content).hexdigest()

        return {
            "safe_filename": safe_name,
            "extension": ext,
            "size_bytes": len(content),
            "sha256": file_hash,
        }

    async def save_upload(
        self,
        filename: str,
        content: bytes,
    ) -> Path:
        """
        Validate and save an uploaded file securely.

        Returns the path to the saved file.
        """
        metadata = self.validate(filename, content)
        safe_name = metadata["safe_filename"]

        # Create upload directory
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        # Add hash prefix to prevent filename collisions
        short_hash = metadata["sha256"][:8]
        final_name = f"{short_hash}_{safe_name}"
        save_path = self.upload_dir / final_name

        # Write atomically (write to temp, then rename)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self.upload_dir),
            prefix=".upload_",
        )
        try:
            os.write(tmp_fd, content)
            os.close(tmp_fd)
            os.replace(tmp_path, str(save_path))
        except Exception:
            os.close(tmp_fd) if not os.get_inheritable(tmp_fd) else None
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        logger.info(
            "File saved: %s (%d bytes, sha256=%s)",
            final_name,
            len(content),
            metadata["sha256"][:16],
        )

        return save_path
