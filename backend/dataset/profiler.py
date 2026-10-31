"""
Automated data profiling engine for user-uploaded datasets.
Calculates distribution metrics, data quality alerts, and RAG text suitability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.dataset.models import ColumnProfile, DataQualityReport

logger = logging.getLogger(__name__)


class DataProfiler:
    """Profiles tabular data for statistical health, quality risks, and schema inference."""

    @classmethod
    def profile_parquet_file(
        cls,
        parquet_path: Path | str,
        max_preview: int = 50,
    ) -> DataQualityReport:
        """
        Scalably profile a Parquet file directly using DuckDB in a consolidated
        single-pass query without loading the dataset into Pandas RAM.
        Optimized for datasets from 10 MB to 5+ GB.
        """
        import duckdb
        path_obj = Path(parquet_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Parquet file not found: {path_obj}")

        posix_path = path_obj.as_posix()
        con = duckdb.connect(":memory:")

        try:
            # 1. Fetch column schema via DESCRIBE (metadata only, zero scan)
            describe_rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{posix_path}')").fetchall()
            total_columns = len(describe_rows)

            if total_columns == 0:
                return DataQualityReport(
                    total_rows=0,
                    total_columns=0,
                    duplicate_rows=0,
                    duplicate_pct=0.0,
                    missing_cells=0,
                    missing_cells_pct=0.0,
                    columns=[],
                    warnings=["Dataset contains zero columns."],
                    preview_rows=[],
                )

            # 2. Build a single consolidated query for row count, duplicates, and column stats
            select_exprs = [
                "COUNT(1) AS _total_rows",
            ]

            # Duplicate detection via row hash
            try:
                select_exprs.append("COUNT(1) - COUNT(DISTINCT hash(t)) AS _duplicate_rows")
            except Exception:
                select_exprs.append("0 AS _duplicate_rows")

            col_meta: list[dict[str, Any]] = []
            for idx, (col_name, col_type, *_) in enumerate(describe_rows):
                escaped_col = '"' + col_name.replace('"', '""') + '"'
                type_upper = col_type.upper()
                is_numeric = any(t in type_upper for t in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "HUGEINT"))
                is_bool = "BOOL" in type_upper
                is_datetime = any(t in type_upper for t in ("DATE", "TIME", "TIMESTAMP"))

                info: dict[str, Any] = {
                    "idx": idx,
                    "name": col_name,
                    "escaped": escaped_col,
                    "type": col_type,
                    "is_numeric": is_numeric,
                    "is_bool": is_bool,
                    "is_datetime": is_datetime,
                }

                # Column null count and approximate distinct count
                select_exprs.append(f"COUNT(1) - COUNT({escaped_col}) AS col_{idx}_nulls")
                select_exprs.append(f"APPROX_COUNT_DISTINCT({escaped_col}) AS col_{idx}_unq")

                if is_numeric:
                    select_exprs.append(f"MIN({escaped_col}) AS col_{idx}_min")
                    select_exprs.append(f"MAX({escaped_col}) AS col_{idx}_max")
                    select_exprs.append(f"AVG(CAST({escaped_col} AS DOUBLE)) AS col_{idx}_avg")
                elif is_datetime:
                    select_exprs.append(f"MIN({escaped_col}) AS col_{idx}_min")
                    select_exprs.append(f"MAX({escaped_col}) AS col_{idx}_max")
                elif not is_bool:
                    # String/text average length
                    select_exprs.append(f"AVG(LENGTH(CAST({escaped_col} AS VARCHAR))) AS col_{idx}_len")

                col_meta.append(info)

            consolidated_sql = f"""
                SELECT {', '.join(select_exprs)}
                FROM read_parquet('{posix_path}') AS t
            """

            try:
                stats_df = con.execute(consolidated_sql).fetchdf()
                stats_row = stats_df.iloc[0].to_dict()
            except Exception as e:
                logger.warning("Consolidated profile query failed, falling back to minimal scan: %s", e)
                # Fallback in case of column name collisions or type cast issues
                total_rows = int(con.execute(f"SELECT COUNT(1) FROM read_parquet('{posix_path}')").fetchone()[0])
                stats_row = {"_total_rows": total_rows, "_duplicate_rows": 0}

            total_rows = int(stats_row.get("_total_rows", 0))
            if total_rows == 0:
                return DataQualityReport(
                    total_rows=0,
                    total_columns=total_columns,
                    duplicate_rows=0,
                    duplicate_pct=0.0,
                    missing_cells=0,
                    missing_cells_pct=0.0,
                    columns=[],
                    warnings=["Dataset contains zero rows."],
                    preview_rows=[],
                )

            duplicate_rows = int(stats_row.get("_duplicate_rows", 0))
            duplicate_pct = round((duplicate_rows / total_rows) * 100, 2) if total_rows > 0 else 0.0

            # 3. Preview rows (bounded to max_preview)
            preview_df = con.execute(f"SELECT * FROM read_parquet('{posix_path}') LIMIT {max_preview}").fetchdf()
            clean_preview = preview_df.astype(object).where(pd.notnull(preview_df), None).to_dict(orient="records")

            column_profiles: list[ColumnProfile] = []
            warnings: list[str] = []
            total_missing_cells = 0

            # 4. Assemble column profiles from the consolidated result dictionary
            for info in col_meta:
                idx = info["idx"]
                col_name = info["name"]
                escaped_col = info["escaped"]
                col_type = info["type"]
                is_numeric = info["is_numeric"]
                is_bool = info["is_bool"]
                is_datetime = info["is_datetime"]
                is_text = False

                null_count = int(stats_row.get(f"col_{idx}_nulls", 0) or 0)
                unique_count = int(stats_row.get(f"col_{idx}_unq", 0) or 0)
                null_pct = round((null_count / total_rows) * 100, 2)
                unique_pct = round((unique_count / total_rows) * 100, 2)
                total_missing_cells += null_count

                min_val = None
                max_val = None
                mean_val = None
                top_values: list[dict[str, Any]] = []

                if is_numeric and null_count < total_rows:
                    mn = stats_row.get(f"col_{idx}_min")
                    mx = stats_row.get(f"col_{idx}_max")
                    av = stats_row.get(f"col_{idx}_avg")
                    min_val = float(mn) if mn is not None and not pd.isna(mn) else None
                    max_val = float(mx) if mx is not None and not pd.isna(mx) else None
                    mean_val = round(float(av), 4) if av is not None and not pd.isna(av) else None

                elif is_datetime and null_count < total_rows:
                    mn = stats_row.get(f"col_{idx}_min")
                    mx = stats_row.get(f"col_{idx}_max")
                    min_val = str(mn) if mn is not None and not pd.isna(mn) else None
                    max_val = str(mx) if mx is not None and not pd.isna(mx) else None

                elif not is_bool and null_count < total_rows:
                    # Check text suitability for RAG from the single-pass average length
                    avg_len = float(stats_row.get(f"col_{idx}_len", 0.0) or 0.0)
                    if avg_len > 45:
                        is_text = True

                    # Only fetch top frequent values for discrete categorical columns (distinct <= 200)
                    if unique_count <= 200 and total_rows > 0:
                        try:
                            # Sample if table is large (> 50,000 rows) to keep latency under 100ms
                            sample_clause = "USING SAMPLE 20000" if total_rows > 50000 else ""
                            top_sql = f"""
                                SELECT CAST({escaped_col} AS VARCHAR) AS val, COUNT(*) AS cnt
                                FROM read_parquet('{posix_path}') {sample_clause}
                                WHERE {escaped_col} IS NOT NULL
                                GROUP BY 1 ORDER BY cnt DESC LIMIT 5
                            """
                            top_res = con.execute(top_sql).fetchall()
                            top_values = [{"value": str(r[0]), "count": int(r[1])} for r in top_res]
                        except Exception:
                            top_values = []

                # Quality warnings
                if null_pct == 100.0:
                    warnings.append(f"Column '{col_name}' is 100% null (completely empty).")
                elif null_pct >= 30.0:
                    warnings.append(f"Column '{col_name}' has high null rate ({null_pct}% missing).")

                if unique_count == 1 and total_rows > 1:
                    warnings.append(f"Column '{col_name}' is constant (only 1 distinct value).")

                profile = ColumnProfile(
                    name=col_name,
                    original_name=col_name,
                    dtype=col_type,
                    null_count=null_count,
                    null_pct=null_pct,
                    unique_count=unique_count,
                    unique_pct=unique_pct,
                    min_val=min_val,
                    max_val=max_val,
                    mean_val=mean_val,
                    top_values=top_values,
                    is_numeric=is_numeric,
                    is_temporal=is_datetime,
                    is_text=is_text,
                )
                column_profiles.append(profile)

            total_cells = total_rows * total_columns
            missing_cells_pct = round((total_missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

            return DataQualityReport(
                total_rows=total_rows,
                total_columns=total_columns,
                duplicate_rows=duplicate_rows,
                duplicate_pct=duplicate_pct,
                missing_cells=total_missing_cells,
                missing_cells_pct=missing_cells_pct,
                columns=column_profiles,
                warnings=warnings,
                preview_rows=clean_preview,
            )

        finally:
            con.close()

    @staticmethod
    def profile_dataframe(df: pd.DataFrame, max_preview: int = 100) -> DataQualityReport:
        """
        Generate a comprehensive data quality and profiling report for a DataFrame.
        """
        total_rows = len(df)
        total_columns = len(df.columns)

        if total_rows == 0 or total_columns == 0:
            return DataQualityReport(
                total_rows=total_rows,
                total_columns=total_columns,
                duplicate_rows=0,
                duplicate_pct=0.0,
                missing_cells=0,
                missing_cells_pct=0.0,
                columns=[],
                warnings=["Dataset contains zero rows or columns."],
                preview_rows=[],
            )

        # 1. Row-level duplication
        try:
            duplicate_count = int(df.duplicated().sum())
        except Exception:
            duplicate_count = 0
        duplicate_pct = round((duplicate_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

        # 2. Total missingness
        total_cells = total_rows * total_columns
        try:
            total_missing = int(df.isna().sum().sum())
        except Exception:
            total_missing = 0
        missing_cells_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0

        column_profiles: list[ColumnProfile] = []
        warnings: list[str] = []

        if duplicate_pct > 10.0:
            warnings.append(
                f"High duplicate row frequency: {duplicate_count:,} duplicate rows ({duplicate_pct}% of dataset)."
            )

        # 3. Column-by-column profiling
        for col_name in df.columns:
            series = df[col_name]
            col_profile, col_warnings = DataProfiler._profile_column(series, str(col_name), total_rows)
            column_profiles.append(col_profile)
            warnings.extend(col_warnings)

        # 4. Preview rows (safely JSON-serializable)
        preview_df = df.head(max_preview)
        # Convert non-standard types to Python primitives
        clean_preview = preview_df.astype(object).where(pd.notnull(preview_df), None).to_dict(orient="records")

        return DataQualityReport(
            total_rows=total_rows,
            total_columns=total_columns,
            duplicate_rows=duplicate_count,
            duplicate_pct=duplicate_pct,
            missing_cells=total_missing,
            missing_cells_pct=missing_cells_pct,
            columns=column_profiles,
            warnings=warnings,
            preview_rows=clean_preview,
        )

    @staticmethod
    def _profile_column(
        series: pd.Series, col_name: str, total_rows: int
    ) -> tuple[ColumnProfile, list[str]]:
        """Profile an individual column."""
        warnings: list[str] = []
        null_count = int(series.isna().sum())
        null_pct = round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

        non_null_series = series.dropna()
        unique_count = int(non_null_series.nunique()) if len(non_null_series) > 0 else 0
        unique_pct = round((unique_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

        # Data type detection
        is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        is_bool = pd.api.types.is_bool_dtype(series)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)

        # Check if string could be parsed as datetime if object
        if not is_datetime and not is_numeric and not is_bool and len(non_null_series) > 0:
            # Check sample to see if ISO date string
            first_val = str(non_null_series.iloc[0])
            if len(first_val) in (10, 19, 20, 23, 24, 25) and ("-" in first_val or "/" in first_val):
                try:
                    pd.to_datetime(non_null_series.head(20), errors="raise")
                    is_datetime = True
                except Exception:
                    pass

        min_val = None
        max_val = None
        mean_val = None
        top_values: list[dict[str, Any]] = []
        is_text = False

        if is_numeric and len(non_null_series) > 0:
            try:
                min_val = float(non_null_series.min())
                max_val = float(non_null_series.max())
                mean_val = round(float(non_null_series.mean()), 4)
                sql_type = "BIGINT" if pd.api.types.is_integer_dtype(series) else "DOUBLE"
            except Exception:
                sql_type = "DOUBLE"
        elif is_datetime and len(non_null_series) > 0:
            try:
                min_val = str(non_null_series.min())
                max_val = str(non_null_series.max())
            except Exception:
                pass
            sql_type = "TIMESTAMP"
        elif is_bool:
            sql_type = "BOOLEAN"
        else:
            sql_type = "VARCHAR"
            # Unstructured text detection: if average string length > 45 chars, flag for RAG
            if len(non_null_series) > 0:
                avg_len = non_null_series.astype(str).str.len().mean()
                if avg_len > 45:
                    is_text = True

            # Calculate top frequent values for categorical / string columns
            if len(non_null_series) > 0:
                try:
                    val_counts = non_null_series.value_counts().head(5)
                    top_values = [
                        {"value": str(k), "count": int(v)} for k, v in val_counts.items()
                    ]
                except Exception:
                    pass

        # Data Quality Warnings
        if null_pct == 100.0:
            warnings.append(f"Column '{col_name}' is 100% null (completely empty).")
        elif null_pct >= 30.0:
            warnings.append(f"Column '{col_name}' has high null rate ({null_pct}% missing).")

        if unique_count == 1 and total_rows > 1:
            warnings.append(f"Column '{col_name}' is constant (only 1 distinct value).")

        profile = ColumnProfile(
            name=col_name,
            original_name=col_name,
            dtype=sql_type,
            null_count=null_count,
            null_pct=null_pct,
            unique_count=unique_count,
            unique_pct=unique_pct,
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
            top_values=top_values,
            is_numeric=is_numeric,
            is_temporal=is_datetime,
            is_text=is_text,
        )

        return profile, warnings
