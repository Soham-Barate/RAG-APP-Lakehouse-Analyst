"""
Scalable streaming ingestion and bounded-memory Parquet conversion engine.
Eliminates full-file memory allocations: streams uploads directly to disk
and converts large CSV/JSON datasets to Parquet using DuckDB out-of-core engine.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHUNK_SIZE_BYTES = 4 * 1024 * 1024  # 4 MB streaming buffer
DEFAULT_MAX_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB max upload limit
ALLOWED_EXTENSIONS = {".csv", ".tsv", ".parquet", ".json", ".jsonl", ".xlsx", ".xls"}


class StreamingIngestionError(Exception):
    """Raised when streaming ingestion or conversion fails."""
    pass


def sanitize_column_identifier(col: str) -> str:
    """Convert raw column strings into safe, valid SQL identifiers."""
    clean = str(col).strip().lower()
    clean = re.sub(r"[^a-z0-9_]", "_", clean)
    clean = re.sub(r"_+", "_", clean)
    clean = clean.strip("_")
    if not clean:
        clean = "col"
    if clean[0].isdigit():
        clean = f"col_{clean}"
    return clean


async def stream_upload_file(
    file_chunks: AsyncGenerator[bytes, None],
    destination_path: Path,
    max_size_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    chunk_callback: Callable[[int, int], None] | None = None,
) -> int:
    """
    Stream incoming file chunks directly to disk with bounded memory.
    Enforces maximum size limit and cleans up on failure.

    Args:
        file_chunks: Async iterator of byte chunks.
        destination_path: Target disk path.
        max_size_bytes: Maximum allowed bytes before aborting.
        chunk_callback: Optional callback(bytes_written, total_bytes) for progress.

    Returns:
        Total bytes written.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination_path.with_suffix(f".upload_{uuid.uuid4().hex[:6]}.tmp")
    bytes_written = 0

    try:
        with open(temp_path, "wb") as f:
            async for chunk in file_chunks:
                if not chunk:
                    continue
                bytes_written += len(chunk)
                if bytes_written > max_size_bytes:
                    raise StreamingIngestionError(
                        f"File size exceeds maximum limit of {max_size_bytes / (1024 * 1024):.1f} MB"
                    )
                f.write(chunk)
                if chunk_callback:
                    chunk_callback(len(chunk), bytes_written)

        if bytes_written == 0:
            raise StreamingIngestionError("Uploaded file is empty (0 bytes).")

        # Atomic rename once complete
        if destination_path.exists():
            destination_path.unlink()
        temp_path.replace(destination_path)
        logger.info("Successfully streamed %d bytes to %s", bytes_written, destination_path)
        return bytes_written

    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


def convert_to_silver_parquet(
    source_path: Path,
    target_parquet_path: Path,
    max_memory_mb: int = 2048,
    row_group_size: int = 100_000,
) -> dict[str, Any]:
    """
    Convert raw Bronze file (CSV, TSV, Parquet, JSON) into a cleaned,
    column-sanitized, Snappy-compressed Silver Parquet dataset using
    DuckDB out-of-core streaming without loading full dataset into RAM.

    Returns:
        Metadata dictionary with row_count, column_count, schema, and columns.
    """
    ext = source_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise StreamingIngestionError(f"Unsupported format '{ext}'.")

    target_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target_parquet_path.with_suffix(".tmp.parquet")

    # Use in-process DuckDB with bounded memory for zero-copy streaming
    con = duckdb.connect(
        database=":memory:",
        config={
            "max_memory": f"{max_memory_mb}MB",
            "threads": min(os.cpu_count() or 4, 8),
            "preserve_insertion_order": False,
        },
    )

    try:
        source_posix = source_path.as_posix()
        temp_target_posix = temp_target.as_posix()

        # Step 1: Create a view over the raw input using DuckDB's streaming readers
        if ext in (".csv", ".tsv"):
            delimiter = "\\t" if ext == ".tsv" else ","
            con.execute(f"""
                CREATE VIEW raw_input AS 
                SELECT * FROM read_csv_auto(
                    '{source_posix}',
                    delim='{delimiter}',
                    header=true,
                    sample_size=20000,
                    ignore_errors=false
                );
            """)
        elif ext == ".parquet":
            con.execute(f"""
                CREATE VIEW raw_input AS 
                SELECT * FROM read_parquet('{source_posix}');
            """)
        elif ext in (".json", ".jsonl"):
            con.execute(f"""
                CREATE VIEW raw_input AS 
                SELECT * FROM read_json_auto(
                    '{source_posix}',
                    format='auto',
                    sample_size=20000
                );
            """)
        elif ext in (".xlsx", ".xls"):
            # Excel requires pandas parsing (bounded for small files)
            import pandas as pd
            df = pd.read_excel(source_path)
            con.register("raw_input", df)

        # Step 2: Extract and sanitize schema columns
        schema_info = con.execute("DESCRIBE raw_input;").fetchall()
        # schema_info: [(column_name, column_type, null, key, default, extra), ...]

        select_expressions: list[str] = []
        sanitized_columns: list[str] = []
        used_names: set[str] = set()

        for col_name, col_type, *_ in schema_info:
            base = sanitize_column_identifier(col_name)
            candidate = base
            counter = 2
            while candidate in used_names:
                candidate = f"{base}_{counter}"
                counter += 1
            used_names.add(candidate)

            # Escape raw column name with double quotes
            escaped_raw = '"' + col_name.replace('"', '""') + '"'
            select_expressions.append(f"{escaped_raw} AS \"{candidate}\"")
            sanitized_columns.append(candidate)

        projection_sql = ", ".join(select_expressions)

        # Step 3: Stream directly into Parquet with Snappy compression and bounded row groups
        copy_sql = f"""
            COPY (SELECT {projection_sql} FROM raw_input)
            TO '{temp_target_posix}'
            (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE {row_group_size});
        """
        con.execute(copy_sql)

        # Step 4: Atomic finalize
        if target_parquet_path.exists():
            target_parquet_path.unlink()
        temp_target.replace(target_parquet_path)

        # Step 5: Read summary statistics from Parquet metadata without full read
        parquet_file = pq.ParquetFile(target_parquet_path)
        total_rows = parquet_file.metadata.num_rows
        total_columns = parquet_file.metadata.num_columns
        parquet_size_bytes = target_parquet_path.stat().st_size

        logger.info(
            "Converted %s -> %s (%d rows, %d cols, %.2f MB)",
            source_path.name,
            target_parquet_path.name,
            total_rows,
            total_columns,
            parquet_size_bytes / (1024 * 1024),
        )

        return {
            "row_count": total_rows,
            "column_count": total_columns,
            "size_bytes": parquet_size_bytes,
            "columns": sanitized_columns,
            "row_groups": parquet_file.metadata.num_row_groups,
        }

    except Exception as exc:
        if temp_target.exists():
            try:
                temp_target.unlink()
            except OSError:
                pass
        logger.error("Streaming Parquet conversion failed for %s: %s", source_path, exc)
        raise StreamingIngestionError(f"Failed converting {source_path.name} to Parquet: {exc}") from exc
    finally:
        con.close()

