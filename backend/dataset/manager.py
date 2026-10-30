"""
Dataset management service for upload handling, format validation,
storage isolation, Parquet transformation, and DuckDB registration.
"""

from __future__ import annotations

import io
import json
import logging
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.dataset.models import DataQualityReport, DatasetMetadata
from backend.dataset.profiler import DataProfiler

logger = logging.getLogger(__name__)

# Constants
DEFAULT_UPLOAD_DIR = Path("data/uploads")
MANIFEST_FILENAME = "manifest.json"
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl"}

# Synthetic lakehouse demo identifier
DEMO_DATASET_ID = "demo_lakehouse"


class DatasetValidationError(Exception):
    """Raised when an uploaded file violates safety or format standards."""
    pass


class DatasetManager:
    """
    Manages user dataset lifecycles: validation, persistent storage,
    Parquet materialization, profiling, and isolated DuckDB registration.
    """

    def __init__(self, upload_dir: str | Path = DEFAULT_UPLOAD_DIR) -> None:
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.upload_dir / MANIFEST_FILENAME
        self._ensure_manifest()

    def _ensure_manifest(self) -> None:
        """Create or validate manifest file on disk."""
        if not self.manifest_path.exists():
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=2)

    def _read_manifest(self) -> dict[str, dict[str, Any]]:
        """Read current manifest from disk."""
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_manifest(self, manifest: dict[str, dict[str, Any]]) -> None:
        """Atomically persist manifest to disk."""
        temp_path = self.manifest_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        temp_path.replace(self.manifest_path)

    @staticmethod
    def sanitize_column_name(col: str) -> str:
        """
        Convert raw column strings into safe, valid SQL identifiers.
        Lowercase, alphanumeric + underscore, no leading digits.
        """
        clean = str(col).strip().lower()
        clean = re.sub(r"[^a-z0-9_]", "_", clean)
        clean = re.sub(r"_+", "_", clean)
        clean = clean.strip("_")
        if not clean:
            clean = "col"
        if clean[0].isdigit():
            clean = f"col_{clean}"
        return clean

    @classmethod
    def sanitize_dataframe_columns(cls, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
        """
        Sanitize all DataFrame column names and return DataFrame with deduplicated columns
        and a mapping of {original_name: sanitized_name}.
        """
        name_map: dict[str, str] = {}
        used_names: set[str] = set()
        new_cols: list[str] = []

        for original in df.columns:
            base = cls.sanitize_column_name(str(original))
            candidate = base
            counter = 2
            while candidate in used_names:
                candidate = f"{base}_{counter}"
                counter += 1
            used_names.add(candidate)
            new_cols.append(candidate)
            name_map[str(original)] = candidate

        df.columns = new_cols
        return df, name_map

    def list_datasets(self) -> list[DatasetMetadata]:
        """
        List all available datasets: the built-in synthetic demo lakehouse
        plus all user-uploaded datasets.
        """
        datasets: list[DatasetMetadata] = []

        # 1. Built-in synthetic demo dataset
        demo_meta = DatasetMetadata(
            dataset_id=DEMO_DATASET_ID,
            name="E-Commerce Lakehouse (Demo)",
            filename="synthetic_lakehouse.parquet",
            file_format="parquet",
            size_bytes=1024 * 1024 * 5,
            row_count=5000,
            column_count=23,
            table_name="products, orders, customers",
            created_at=datetime.now(timezone.utc).isoformat(),
            storage_path="data/gold/",
            parquet_path="data/gold/fact_orders.parquet",
            has_text_indexed=True,
            is_demo=True,
            tables=["orders", "products", "customers"],
        )
        datasets.append(demo_meta)

        # 2. Persisted user datasets from manifest
        manifest = self._read_manifest()
        for ds_id, data in manifest.items():
            try:
                datasets.append(DatasetMetadata(**data))
            except Exception as e:
                logger.warning("Failed parsing metadata for dataset %s: %s", ds_id, e)

        return datasets

    def get_dataset(self, dataset_id: str) -> DatasetMetadata | None:
        """Fetch metadata for a specific dataset ID."""
        for ds in self.list_datasets():
            if ds.dataset_id == dataset_id:
                return ds
        return None

    def get_profiling_report(self, dataset_id: str) -> DataQualityReport | None:
        """Retrieve cached profiling report for a dataset."""
        if dataset_id == DEMO_DATASET_ID:
            # Generate or return demo profile
            return self._get_demo_profiling_report()

        profile_path = self.upload_dir / dataset_id / "profile.json"
        if not profile_path.exists():
            return None

        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                return DataQualityReport(**json.load(f))
        except Exception as e:
            logger.error("Failed loading profiling report for %s: %s", dataset_id, e)
            return None

    def _get_demo_profiling_report(self) -> DataQualityReport:
        """Synthesize profiling report for demo lakehouse tables."""
        return DataQualityReport(
            total_rows=5000,
            total_columns=23,
            duplicate_rows=0,
            duplicate_pct=0.0,
            missing_cells=0,
            missing_cells_pct=0.0,
            columns=[],
            warnings=["Operating on verified synthetic medallion lakehouse (Bronze/Silver/Gold)."],
            preview_rows=[],
        )

    def get_sample_rows(self, dataset_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve top sample rows from the dataset's Parquet file."""
        meta = self.get_dataset(dataset_id)
        if not meta:
            return []

        if meta.is_demo:
            try:
                import duckdb
                con = duckdb.connect(":memory:")
                parquet_path = Path("data/gold/fact_orders.parquet")
                if parquet_path.exists():
                    df = con.execute(f"SELECT * FROM read_parquet('{parquet_path.as_posix()}') LIMIT {limit}").fetchdf()
                    return df.to_dict(orient="records")
            except Exception:
                pass
            return []

        parquet_path = Path(meta.parquet_path)
        if not parquet_path.exists():
            return []

        try:
            import duckdb
            con = duckdb.connect(":memory:")
            df = con.execute(f"SELECT * FROM read_parquet('{parquet_path.as_posix()}') LIMIT {limit}").fetchdf()
            return df.to_dict(orient="records")
        except Exception:
            try:
                df = pd.read_parquet(parquet_path)
                return df.head(limit).to_dict(orient="records")
            except Exception:
                return []

    def upload_dataset(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str | None = None,
    ) -> DatasetMetadata:
        """
        Validate, parse, profile, and persist an uploaded dataset file.
        Supports CSV, XLSX, XLS, Parquet, JSON, and JSONL.
        """
        # 1. Size Validation
        size_bytes = len(file_bytes)
        if size_bytes == 0:
            raise DatasetValidationError("Uploaded file is empty (0 bytes).")
        if size_bytes > MAX_UPLOAD_SIZE_BYTES:
            raise DatasetValidationError(
                f"File size ({size_bytes / (1024*1024):.1f} MB) exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB."
            )

        # 2. Extension & Magic Bytes Validation
        file_path_obj = Path(filename)
        ext = file_path_obj.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise DatasetValidationError(
                f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            )

        self._validate_file_signature(file_bytes, ext)

        # 3. Parse File into DataFrame Safely
        df = self._parse_file_to_df(file_bytes, ext, filename)

        if len(df) == 0:
            raise DatasetValidationError("Dataset contains zero data rows.")
        if len(df.columns) == 0:
            raise DatasetValidationError("Dataset contains zero columns.")

        # 4. Sanitize Column Names
        df, _ = self.sanitize_dataframe_columns(df)

        # 5. Create Isolated Dataset Directory
        dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
        dataset_dir = self.upload_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        raw_saved_path = dataset_dir / f"raw{ext}"
        with open(raw_saved_path, "wb") as f:
            f.write(file_bytes)

        parquet_saved_path = dataset_dir / "data.parquet"
        df.to_parquet(parquet_saved_path, index=False)

        # 6. Run Data Profiling & Save Report
        profile_report = DataProfiler.profile_dataframe(df)
        profile_path = dataset_dir / "profile.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            f.write(profile_report.model_dump_json(indent=2))

        # Check for text columns suitable for RAG
        has_text_cols = any(col.is_text for col in profile_report.columns)

        # Derive safe table name
        safe_name = self.sanitize_column_name(file_path_obj.stem)
        table_name = f"user_{safe_name}" if not safe_name.startswith("user_") else safe_name
        # Keep table name unique
        manifest = self._read_manifest()
        existing_tables = {d.get("table_name") for d in manifest.values()}
        if table_name in existing_tables:
            table_name = f"{table_name}_{dataset_id[-4:]}"

        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            name=file_path_obj.name,
            filename=filename,
            file_format=ext.lstrip("."),
            size_bytes=size_bytes,
            row_count=len(df),
            column_count=len(df.columns),
            table_name=table_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            storage_path=raw_saved_path.as_posix(),
            parquet_path=parquet_saved_path.as_posix(),
            has_text_indexed=has_text_cols,
            is_demo=False,
            tables=[table_name],
        )

        manifest[dataset_id] = metadata.model_dump()
        self._write_manifest(manifest)

        logger.info(
            "Successfully ingested user dataset '%s' (id=%s, rows=%d, cols=%d, table=%s)",
            filename,
            dataset_id,
            len(df),
            len(df.columns),
            table_name,
        )

        return metadata

    def delete_dataset(self, dataset_id: str) -> bool:
        """Permanently delete an uploaded dataset and purge its artifacts."""
        if dataset_id == DEMO_DATASET_ID:
            raise DatasetValidationError("Cannot delete the core synthetic demo lakehouse dataset.")

        manifest = self._read_manifest()
        if dataset_id not in manifest:
            return False

        # Remove directory
        dataset_dir = self.upload_dir / dataset_id
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir, ignore_errors=True)

        del manifest[dataset_id]
        self._write_manifest(manifest)
        logger.info("Deleted dataset %s and cleaned up storage.", dataset_id)
        return True

    def register_all_in_duckdb(self, duckdb_conn: Any) -> None:
        """Register all user datasets in DuckDB using in-memory DataFrame registration."""
        manifest = self._read_manifest()
        for ds_id, data in manifest.items():
            try:
                parquet_path = Path(data.get("parquet_path", ""))
                table_name = data.get("table_name", f"tbl_{ds_id}")
                if parquet_path.exists():
                    df = pd.read_parquet(parquet_path)
                    duckdb_conn.register(table_name, df)
            except Exception as e:
                logger.error("Failed registering dataset %s into DuckDB: %s", ds_id, e)

    def get_schema_for_sql(self, dataset_id: str) -> dict[str, set[str]]:
        """
        Return the table names and allowed column names for a dataset.
        Enforces strict dataset isolation.
        """
        meta = self.get_dataset(dataset_id)
        if not meta or meta.is_demo:
            from backend.rag.nodes.sql_node import ALLOWED_COLUMNS
            return ALLOWED_COLUMNS

        # For user dataset, read the columns from the profile or parquet
        profile = self.get_profiling_report(dataset_id)
        if profile and profile.columns:
            columns = {col.name for col in profile.columns}
        else:
            try:
                import duckdb
                con = duckdb.connect(":memory:")
                df = con.execute(f"SELECT * FROM read_parquet('{meta.parquet_path}') LIMIT 0").fetchdf()
                columns = set(df.columns)
            except Exception:
                columns = set()

        return {meta.table_name: columns}

    def get_schema_prompt_for_dataset(self, dataset_id: str) -> tuple[str, str, set[str]]:
        """
        Generate dynamic LLM schema instructions for Text-to-SQL.
        Returns:
            (schema_prompt_string, primary_table_name, allowed_columns_set)
        """
        meta = self.get_dataset(dataset_id)
        if not meta or meta.is_demo:
            from backend.rag.nodes.sql_node import ALLOWED_COLUMNS
            from backend.rag.prompts.templates import TEXT_TO_SQL_PROMPT
            return TEXT_TO_SQL_PROMPT, "orders", set().union(*ALLOWED_COLUMNS.values())

        profile = self.get_profiling_report(dataset_id)
        col_descriptions: list[str] = []
        allowed_cols: set[str] = set()

        if profile:
            for col in profile.columns:
                allowed_cols.add(col.name)
                note = f" (Sample: {col.top_values[0]['value']})" if col.top_values else ""
                col_descriptions.append(f"  - {col.name} ({col.dtype}){note}")
        else:
            schema_map = self.get_schema_for_sql(dataset_id)
            for col in schema_map.get(meta.table_name, set()):
                allowed_cols.add(col)
                col_descriptions.append(f"  - {col} (VARCHAR)")

        cols_str = "\n".join(col_descriptions)

        custom_prompt = (
            f"You are an enterprise lakehouse Text-to-SQL analyst.\n"
            f"Translate the user's natural language question into safe, read-only DuckDB SQL.\n\n"
            f"Active Table: {meta.table_name} (Dataset: {meta.name}, {meta.row_count:,} rows)\n"
            f"Available Columns:\n"
            f"{cols_str}\n\n"
            f"Strict Safety and Execution Constraints:\n"
            f"1. Generate ONLY a single SELECT query referencing table '{meta.table_name}'.\n"
            f"2. You MUST NOT reference any other tables or datasets.\n"
            f"3. Do NOT use DDL, DML, file-reading functions (read_csv, read_parquet), or system commands.\n"
            f"4. Always include a sensible LIMIT (default 50) unless computing an aggregate (SUM, AVG, COUNT).\n"
            f"5. Wrap column names in double quotes if they conflict with SQL keywords.\n"
            f"6. Return ONLY the raw SQL query inside ```sql codeblocks or as clean SQL text."
        )

        return custom_prompt, meta.table_name, allowed_cols

    @staticmethod
    def _validate_file_signature(file_bytes: bytes, ext: str) -> None:
        """Validate magic bytes / file signatures to prevent disguised executables."""
        if ext == ".parquet":
            if not file_bytes.startswith(b"PAR1"):
                raise DatasetValidationError("Invalid Parquet file: Missing 'PAR1' magic header.")
        elif ext in (".xlsx", ".xls"):
            # XLSX is a ZIP container starting with PK\x03\x04
            if ext == ".xlsx" and not file_bytes.startswith(b"PK\x03\x04"):
                raise DatasetValidationError("Invalid XLSX spreadsheet: Corrupted ZIP signature.")
        elif ext in (".json", ".jsonl"):
            # Check for JSON opening
            trimmed = file_bytes.strip()
            if not (trimmed.startswith(b"{") or trimmed.startswith(b"[") or b"\n" in trimmed):
                raise DatasetValidationError("Invalid JSON/JSONL format: Expected JSON object or array.")

    @staticmethod
    def _parse_file_to_df(file_bytes: bytes, ext: str, filename: str) -> pd.DataFrame:
        """Parse raw file bytes into a clean pandas DataFrame."""
        buf = io.BytesIO(file_bytes)
        try:
            if ext == ".csv":
                try:
                    return pd.read_csv(buf, encoding="utf-8")
                except UnicodeDecodeError:
                    buf.seek(0)
                    return pd.read_csv(buf, encoding="latin1")
            elif ext in (".xlsx", ".xls"):
                return pd.read_excel(buf)
            elif ext == ".parquet":
                return pd.read_parquet(buf)
            elif ext in (".json", ".jsonl"):
                try:
                    buf.seek(0)
                    return pd.read_json(buf)
                except ValueError:
                    buf.seek(0)
                    return pd.read_json(buf, lines=True)
            else:
                raise DatasetValidationError(f"Cannot parse file with extension '{ext}'.")
        except DatasetValidationError:
            raise
        except Exception as e:
            raise DatasetValidationError(f"Failed parsing {filename} as {ext}: {str(e)}") from e


# Singleton instance accessor
_default_manager: DatasetManager | None = None


def get_dataset_manager() -> DatasetManager:
    """Retrieve singleton DatasetManager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = DatasetManager()
    return _default_manager
