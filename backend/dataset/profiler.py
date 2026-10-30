"""
Automated data profiling engine for user-uploaded datasets.
Calculates distribution metrics, data quality alerts, and RAG text suitability.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from backend.dataset.models import ColumnProfile, DataQualityReport

logger = logging.getLogger(__name__)


class DataProfiler:
    """Profiles tabular data for statistical health, quality risks, and schema inference."""

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
