"""
API integration tests for dataset management endpoints (/api/v1/datasets).
"""

from __future__ import annotations

import io
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client():
    """Create test client with authenticated bypass."""
    from backend.main import create_app
    from backend.api.middleware.auth import verify_api_key

    app = create_app()
    app.dependency_overrides[verify_api_key] = lambda: "test-auth-key"
    with TestClient(app, headers={"X-API-Key": "test-auth-key"}) as client:
        yield client


def test_list_datasets_endpoint(api_client: TestClient):
    """GET /api/v1/datasets returns available datasets list."""
    response = api_client.get("/api/v1/datasets")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    # Check that demo dataset is present
    assert any(d["dataset_id"] == "demo_lakehouse" for d in data)


def test_upload_and_delete_dataset_flow(api_client: TestClient):
    """Full lifecycle: upload CSV -> inspect detail -> preview -> delete."""
    from config.settings import get_settings
    active_key = get_settings().api_keys[0] if get_settings().api_keys else "dev-key-1"

    csv_bytes = b"user_id,plan,mrr\nU101,Pro,99.0\nU102,Enterprise,499.0\n"
    files = {"file": ("saas_subscriptions.csv", io.BytesIO(csv_bytes), "text/csv")}
    headers = {"X-API-Key": active_key}

    # 1. Upload
    upload_resp = api_client.post("/api/v1/datasets/upload", files=files, headers=headers)
    assert upload_resp.status_code == 200, f"Upload failed: {upload_resp.text}"
    upload_data = upload_resp.json()
    assert upload_data["success"] is True
    metadata = upload_data["metadata"]
    ds_id = metadata["dataset_id"]
    assert metadata["row_count"] == 2
    assert metadata["column_count"] == 3

    # 2. Get Detail
    detail_resp = api_client.get(f"/api/v1/datasets/{ds_id}")
    assert detail_resp.status_code == 200
    detail_data = detail_resp.json()
    assert detail_data["metadata"]["dataset_id"] == ds_id
    assert detail_data["profile"]["total_rows"] == 2

    # 3. Preview
    prev_resp = api_client.get(f"/api/v1/datasets/{ds_id}/preview?limit=10")
    assert prev_resp.status_code == 200
    prev_data = prev_resp.json()
    assert len(prev_data) == 2

    # 4. Delete
    del_resp = api_client.delete(f"/api/v1/datasets/{ds_id}", headers=headers)
    assert del_resp.status_code == 200
    assert del_resp.json()["success"] is True

    # 5. Confirm Deleted
    confirm_resp = api_client.get(f"/api/v1/datasets/{ds_id}")
    assert confirm_resp.status_code == 404


def test_chunked_upload_flow(api_client: TestClient):
    """Verify genuine chunked streaming upload lifecycle (init -> chunk -> finalize)."""
    import hashlib
    content = b"sensor_id,temperature,pressure\nS1,24.5,101.3\nS2,26.1,101.5\nS3,23.9,101.1\n"
    sha256_hash = hashlib.sha256(content).hexdigest()
    chunk_size = 20

    # 1. Init
    init_res = api_client.post(
        "/api/v1/datasets/upload/init",
        json={
            "filename": "sensors.csv",
            "total_bytes": len(content),
            "chunk_size": chunk_size,
            "checksum_sha256": sha256_hash,
        },
    )
    assert init_res.status_code == 200, f"Init failed: {init_res.text}"
    upload_id = init_res.json()["upload_id"]

    # 2. Upload in chunks
    offset = 0
    chunk_idx = 0
    while offset < len(content):
        chunk = content[offset : offset + chunk_size]
        chunk_res = api_client.post(
            "/api/v1/datasets/upload/chunk",
            data={"upload_id": upload_id, "chunk_index": str(chunk_idx)},
            files={"file": (f"chunk_{chunk_idx}.bin", io.BytesIO(chunk), "application/octet-stream")},
        )
        assert chunk_res.status_code == 200, f"Chunk {chunk_idx} failed: {chunk_res.text}"
        offset += len(chunk)
        chunk_idx += 1

    # 3. Finalize
    fin_res = api_client.post(
        "/api/v1/datasets/upload/finalize",
        json={"upload_id": upload_id},
    )
    assert fin_res.status_code == 200, f"Finalize failed: {fin_res.text}"
    fin_data = fin_res.json()
    assert fin_data["success"] is True
    assert fin_data["metadata"]["row_count"] == 3
    assert fin_data["sha256"] == sha256_hash

    # 4. Clean up
    ds_id = fin_data["metadata"]["dataset_id"]
    api_client.delete(f"/api/v1/datasets/{ds_id}")
