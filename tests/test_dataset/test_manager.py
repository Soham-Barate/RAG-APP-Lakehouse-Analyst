"""
Unit tests for DatasetManager: upload validation, persistence, format handling, and isolation.
"""

import io
import json
import pytest
import pandas as pd
from pathlib import Path

from backend.dataset.manager import (
    DEMO_DATASET_ID,
    DatasetManager,
    DatasetValidationError,
)


@pytest.fixture
def temp_upload_dir(tmp_path: Path) -> Path:
    """Provide an isolated upload directory for tests."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


@pytest.fixture
def dataset_manager(temp_upload_dir: Path) -> DatasetManager:
    """Create a DatasetManager instance pointing to temp directory."""
    return DatasetManager(upload_dir=temp_upload_dir)


class TestDatasetManager:
    """Tests for dataset storage, format parsing, and isolation."""

    def test_list_datasets_includes_demo_by_default(self, dataset_manager: DatasetManager):
        """Manager always presents the synthetic demo lakehouse dataset."""
        datasets = dataset_manager.list_datasets()
        assert len(datasets) >= 1
        demo = next((d for d in datasets if d.dataset_id == DEMO_DATASET_ID), None)
        assert demo is not None
        assert demo.is_demo is True
        assert "orders" in demo.tables

    def test_upload_valid_csv(self, dataset_manager: DatasetManager):
        """CSV upload validates, converts to Parquet, and persists."""
        csv_content = b"customer_id,transaction_val,location\nC001,150.50,NY\nC002,230.00,CA\n"
        meta = dataset_manager.upload_dataset(
            file_bytes=csv_content,
            filename="user_sales.csv",
        )

        assert meta.dataset_id.startswith("ds_")
        assert meta.row_count == 2
        assert meta.column_count == 3
        assert Path(meta.parquet_path).exists()
        assert meta.table_name.startswith("user_")

        # Verify listed
        all_ds = dataset_manager.list_datasets()
        assert any(d.dataset_id == meta.dataset_id for d in all_ds)

    def test_upload_valid_parquet(self, dataset_manager: DatasetManager):
        """Parquet upload preserves schema and creates DuckDB view."""
        df = pd.DataFrame({
            "sku_code": ["A101", "B202"],
            "unit_price": [12.99, 45.50],
            "in_stock": [True, False],
        })
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        parquet_bytes = buf.getvalue()

        meta = dataset_manager.upload_dataset(
            file_bytes=parquet_bytes,
            filename="catalog.parquet",
        )
        assert meta.row_count == 2
        assert meta.column_count == 3

    def test_upload_valid_json(self, dataset_manager: DatasetManager):
        """JSON records upload cleanly."""
        data = [
            {"product": "Laptop", "price": 999.99, "stock": 15},
            {"product": "Mouse", "price": 29.99, "stock": 50},
        ]
        json_bytes = json.dumps(data).encode("utf-8")
        meta = dataset_manager.upload_dataset(
            file_bytes=json_bytes,
            filename="inventory.json",
        )
        assert meta.row_count == 2
        assert meta.column_count == 3

    def test_upload_empty_file_rejected(self, dataset_manager: DatasetManager):
        """Empty 0-byte file raises validation error."""
        with pytest.raises(DatasetValidationError, match="empty"):
            dataset_manager.upload_dataset(b"", "empty.csv")

    def test_upload_unsupported_extension_rejected(self, dataset_manager: DatasetManager):
        """Executable or unsupported extensions are rejected."""
        with pytest.raises(DatasetValidationError, match="Unsupported file format"):
            dataset_manager.upload_dataset(b"echo 1", "script.sh")

    def test_column_sanitization(self, dataset_manager: DatasetManager):
        """Special characters and spaces in column names are safely normalized."""
        csv_content = b"1st Quarter ($),Customer - Name,Total%Return\n100,Alice,5.2\n"
        meta = dataset_manager.upload_dataset(csv_content, "dirty_cols.csv")

        profile = dataset_manager.get_profiling_report(meta.dataset_id)
        assert profile is not None
        col_names = [c.name for c in profile.columns]

        # Ensure no leading digits, no spaces, no special characters
        for name in col_names:
            assert not name[0].isdigit()
            assert " " not in name
            assert "$" not in name
            assert "%" not in name

    def test_delete_user_dataset(self, dataset_manager: DatasetManager):
        """User dataset deletion cleans up directory and manifest."""
        csv_content = b"col1,col2\n1,2\n"
        meta = dataset_manager.upload_dataset(csv_content, "to_delete.csv")
        ds_id = meta.dataset_id

        assert dataset_manager.get_dataset(ds_id) is not None
        success = dataset_manager.delete_dataset(ds_id)
        assert success is True
        assert dataset_manager.get_dataset(ds_id) is None

    def test_cannot_delete_demo_dataset(self, dataset_manager: DatasetManager):
        """Attempting to delete core synthetic demo lakehouse is blocked."""
        with pytest.raises(DatasetValidationError, match="Cannot delete"):
            dataset_manager.delete_dataset(DEMO_DATASET_ID)
