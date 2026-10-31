"""
Unit tests for DataProfiler statistical calculations and anomaly alerts.
"""

import pandas as pd
from backend.dataset.profiler import DataProfiler


def test_profiler_detects_missingness_and_duplicates():
    """Verify null count, missing percentage, and duplicate counts."""
    df = pd.DataFrame({
        "id": [1, 2, 2, 4, 5],
        "category": ["A", "B", "B", None, "A"],
        "metric": [10.0, None, None, 40.0, 50.0],
    })

    report = DataProfiler.profile_dataframe(df)
    assert report.total_rows == 5
    assert report.total_columns == 3
    assert report.duplicate_rows == 1
    assert report.duplicate_pct == 20.0
    assert report.missing_cells == 3

    # Check metric column
    metric_col = next(c for c in report.columns if c.name == "metric")
    assert metric_col.null_count == 2
    assert metric_col.null_pct == 40.0
    assert metric_col.min_val == 10.0
    assert metric_col.max_val == 50.0
    assert metric_col.is_numeric is True


def test_profiler_identifies_text_columns_for_rag():
    """Columns with long prose are flagged as text for vector indexing."""
    df = pd.DataFrame({
        "id": [1, 2],
        "short_code": ["ABC", "XYZ"],
        "review_text": [
            "This enterprise analytics platform provides high-throughput lakehouse queries with extreme reliability.",
            "Customer service was exceptional, resolving our fulfillment issue within twenty minutes of first contact.",
        ],
    })

    report = DataProfiler.profile_dataframe(df)
    review_col = next(c for c in report.columns if c.name == "review_text")
    assert review_col.is_text is True

    short_col = next(c for c in report.columns if c.name == "short_code")
    assert short_col.is_text is False


def test_profiler_warns_on_high_nulls_and_constants():
    """Data quality warnings generated for empty columns and constants."""
    df = pd.DataFrame({
        "constant_val": ["SAME", "SAME", "SAME"],
        "empty_val": [None, None, None],
        "normal": [1, 2, 3],
    })

    report = DataProfiler.profile_dataframe(df)
    assert any("constant" in w.lower() for w in report.warnings)
    assert any("empty" in w.lower() or "100%" in w.lower() for w in report.warnings)


def test_profiler_parquet_consolidated_scan_and_duplicates(tmp_path):
    """Verify DuckDB single-pass consolidated scan on Parquet detects duplicates and stats."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        "id": [1, 2, 2, 4],
        "category": ["X", "Y", "Y", "Z"],
        "price": [10.5, 20.0, 20.0, 45.0],
    })
    parquet_file = tmp_path / "test_data.parquet"
    pq.write_table(table, parquet_file)

    report = DataProfiler.profile_parquet_file(parquet_file)
    assert report.total_rows == 4
    assert report.total_columns == 3
    assert report.duplicate_rows == 1
    assert report.duplicate_pct == 25.0
    assert report.missing_cells == 0

    price_col = next(c for c in report.columns if c.name == "price")
    assert price_col.min_val == 10.5
    assert price_col.max_val == 45.0
    assert price_col.null_count == 0
    assert price_col.is_numeric is True

