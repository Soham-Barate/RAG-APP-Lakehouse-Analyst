"""
Dataset package for user dataset ingestion, profiling, and DuckDB registration.
"""

from backend.dataset.models import ColumnProfile, DataQualityReport, DatasetMetadata
from backend.dataset.profiler import DataProfiler
from backend.dataset.manager import DatasetManager, get_dataset_manager, DEMO_DATASET_ID

__all__ = [
    "ColumnProfile",
    "DataQualityReport",
    "DatasetMetadata",
    "DataProfiler",
    "DatasetManager",
    "get_dataset_manager",
    "DEMO_DATASET_ID",
]
