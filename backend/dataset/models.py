"""
Dataset models for user uploads, profiling reports, and schema isolation.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    """Statistical profile of a single dataset column."""

    name: str = Field(description="Column name (sanitized for SQL safety)")
    original_name: str = Field(description="Original column name as uploaded")
    dtype: str = Field(description="Inferred SQL data type (e.g., VARCHAR, BIGINT, DOUBLE, TIMESTAMP, BOOLEAN)")
    null_count: int = Field(default=0, description="Count of missing or null values")
    null_pct: float = Field(default=0.0, description="Percentage of null values (0.0 - 100.0)")
    unique_count: int = Field(default=0, description="Count of distinct unique values")
    unique_pct: float = Field(default=0.0, description="Percentage of unique values (0.0 - 100.0)")
    min_val: Any | None = Field(default=None, description="Minimum value for numeric/temporal columns")
    max_val: Any | None = Field(default=None, description="Maximum value for numeric/temporal columns")
    mean_val: float | None = Field(default=None, description="Arithmetic mean for numeric columns")
    top_values: list[dict[str, Any]] = Field(default_factory=list, description="Top 5 frequent values with counts")
    is_numeric: bool = Field(default=False, description="Whether column is numeric")
    is_temporal: bool = Field(default=False, description="Whether column is a date/timestamp")
    is_text: bool = Field(default=False, description="Whether column contains unstructured text suitable for RAG")


class DataQualityReport(BaseModel):
    """Comprehensive data quality and profiling audit of an uploaded dataset."""

    total_rows: int = Field(description="Total number of rows")
    total_columns: int = Field(description="Total number of columns")
    duplicate_rows: int = Field(default=0, description="Count of identical duplicate rows")
    duplicate_pct: float = Field(default=0.0, description="Percentage of duplicate rows")
    missing_cells: int = Field(default=0, description="Total missing cells across entire table")
    missing_cells_pct: float = Field(default=0.0, description="Percentage of missing cells across entire table")
    columns: list[ColumnProfile] = Field(default_factory=list, description="Column profiles")
    warnings: list[str] = Field(default_factory=list, description="Data quality risk flags and recommendations")
    preview_rows: list[dict[str, Any]] = Field(default_factory=list, description="First sample rows for inspection")


class DatasetMetadata(BaseModel):
    """Metadata record for a registered lakehouse dataset."""

    dataset_id: str = Field(description="Unique deterministic or random identifier")
    name: str = Field(description="Human-readable dataset display name")
    filename: str = Field(description="Original uploaded file name")
    file_format: str = Field(description="File format extension: csv, xlsx, xls, parquet, json")
    size_bytes: int = Field(default=0, description="File size in bytes")
    row_count: int = Field(default=0, description="Total row count")
    column_count: int = Field(default=0, description="Total column count")
    table_name: str = Field(description="Sanitized DuckDB table/view name (e.g. user_sales_q3)")
    created_at: str = Field(description="ISO 8601 creation timestamp")
    storage_path: str = Field(description="Path to stored raw file")
    parquet_path: str = Field(description="Path to optimized Parquet file")
    bronze_path: str = Field(default="", description="Path to Bronze raw layer storage")
    silver_path: str = Field(default="", description="Path to Silver cleaned Parquet storage")
    gold_path: str = Field(default="", description="Path to Gold aggregated/partitioned Parquet storage")
    processing_duration_sec: float = Field(default=0.0, description="Processing and ingestion duration in seconds")
    status: str = Field(default="ready", description="Dataset state: pending, processing, ready, failed")
    data_quality_score: float = Field(default=100.0, description="Calculated data quality health score (0-100)")
    has_text_indexed: bool = Field(default=False, description="Whether unstructured text was indexed into vector store")
    is_demo: bool = Field(default=False, description="Whether this is the synthetic e-commerce demo dataset")
    tables: list[str] = Field(default_factory=list, description="List of accessible table names in DuckDB")
