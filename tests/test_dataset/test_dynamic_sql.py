"""
Integration tests for SafeSQLNode with user-uploaded datasets.
Verifies dynamic schema enforcement and cross-dataset isolation.
"""

from unittest.mock import AsyncMock
import pytest
from backend.dataset.manager import get_dataset_manager
from backend.rag.nodes.sql_node import SafeSQLNode, SQLValidationError


@pytest.fixture
def mock_llm():
    """Mock LLM generating predefined SQL responses."""
    llm = AsyncMock()
    return llm


@pytest.mark.asyncio
async def test_safe_sql_node_allows_user_dataset_table(mock_llm, tmp_path):
    """User dataset query succeeds when referencing the correct uploaded table."""
    mgr = get_dataset_manager()
    csv_bytes = b"user_id,amount,region\nU1,100,US\nU2,250,EU\n"
    meta = mgr.upload_dataset(csv_bytes, "finance_data.csv")

    node = SafeSQLNode(llm_service=mock_llm)

    # Mock LLM returning valid SQL for the uploaded dataset table
    mock_llm.generate.return_value = (
        f'{{"sql": "SELECT region, SUM(amount) AS total FROM {meta.table_name} GROUP BY region", '
        f'"explanation": "Sum revenue by region"}}'
    )

    state = {
        "query": "What is total revenue by region?",
        "dataset_id": meta.dataset_id,
        "fast_mode": True,
    }

    result = await node(state)
    assert result["sql_error"] is None
    assert len(result["sql_results"]) == 2
    assert any(r.get("region") == "US" for r in result["sql_results"])

    # Clean up
    mgr.delete_dataset(meta.dataset_id)


@pytest.mark.asyncio
async def test_safe_sql_node_blocks_cross_dataset_access(mock_llm):
    """Query on Dataset A cannot access Dataset B or demo tables."""
    mgr = get_dataset_manager()
    csv_a = b"metric_a,val_a\n1,10\n"
    meta_a = mgr.upload_dataset(csv_a, "dataset_a.csv")

    node = SafeSQLNode(llm_service=mock_llm)

    # Attempt to query orders (demo table) while active dataset is dataset_a
    mock_llm.generate.return_value = (
        '{"sql": "SELECT order_id FROM orders LIMIT 10", "explanation": "Leak attempt"}'
    )

    state = {
        "query": "Show me orders",
        "dataset_id": meta_a.dataset_id,
        "fast_mode": True,
    }

    result = await node(state)
    # Validation should reject 'orders' as it is not the active table for dataset_a
    assert result["sql_error"] is not None
    assert "not permitted" in result["sql_error"].lower()

    # Clean up
    mgr.delete_dataset(meta_a.dataset_id)
