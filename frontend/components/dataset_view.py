"""
Dataset management and automated data profiling component for Streamlit frontend.
Provides drag-and-drop file upload, quality auditing, column statistics, and active dataset switching.
"""

from __future__ import annotations

import html
import io
import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev-key-1")


def fetch_datasets() -> list[dict[str, Any]]:
    """Fetch all available datasets from backend API."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/api/v1/datasets", timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    # Fallback to demo dataset
    return [{
        "dataset_id": "demo_lakehouse",
        "name": "E-Commerce Lakehouse (Demo)",
        "filename": "synthetic_lakehouse.parquet",
        "file_format": "parquet",
        "row_count": 5000,
        "column_count": 23,
        "table_name": "products, orders, customers",
        "is_demo": True,
        "tables": ["orders", "products", "customers"],
        "has_text_indexed": True,
    }]


def fetch_dataset_detail(dataset_id: str) -> dict[str, Any] | None:
    """Fetch complete metadata, profiling report, and sample rows for a dataset."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/api/v1/datasets/{dataset_id}", timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def fetch_jobs(limit: int = 5) -> list[dict[str, Any]]:
    """Fetch recent asynchronous dataset jobs from backend API."""
    try:
        resp = httpx.get(f"{BACKEND_URL}/api/v1/datasets/jobs?limit={limit}", timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def upload_dataset_file(uploaded_file: Any) -> tuple[bool, str, dict[str, Any] | None]:
    """Upload a file to the backend dataset ingestion endpoint."""
    try:
        file_bytes = uploaded_file.getvalue()
        files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type or "application/octet-stream")}
        headers = {"X-API-Key": API_KEY}

        resp = httpx.post(
            f"{BACKEND_URL}/api/v1/datasets/upload",
            files=files,
            headers=headers,
            timeout=300.0,
        )

        if resp.status_code == 200:
            data = resp.json()
            return True, "Dataset uploaded, converted to Silver Parquet, and profiled successfully.", data
        else:
            try:
                err = resp.json()
                msg = err.get("detail", str(err))
            except Exception:
                msg = resp.text or f"Server returned {resp.status_code}"
            return False, f"Upload rejected: {msg}", None
    except Exception as exc:
        return False, f"Upload network error: {str(exc)}", None


def delete_dataset(dataset_id: str) -> tuple[bool, str]:
    """Request backend to delete a user dataset."""
    try:
        headers = {"X-API-Key": API_KEY}
        resp = httpx.delete(
            f"{BACKEND_URL}/api/v1/datasets/{dataset_id}",
            headers=headers,
            timeout=10.0,
        )
        if resp.status_code == 200:
            return True, "Dataset deleted."
        return False, resp.json().get("detail", "Failed deleting dataset.")
    except Exception as exc:
        return False, str(exc)


def render_dataset_manager() -> None:
    """Render the primary Dataset Manager view in Streamlit."""
    banner_html = (
        '<div style="margin-bottom:1.5rem;">'
        '<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.25rem;">'
        '<span class="app-version-pill">STORAGE ENGINE</span>'
        '<span class="app-version-pill">DUCKDB + PARQUET</span>'
        '</div>'
        '<h2 style="font-size:1.25rem; font-weight:600; color:var(--text-primary); margin:0 0 0.25rem 0;">'
        'Lakehouse Dataset Manager & Automated Profiler'
        '</h2>'
        '<p style="font-size:0.8125rem; color:var(--text-secondary); margin:0;">'
        'Upload real-world tabular data (CSV, XLSX, Parquet, JSON), inspect data quality distribution metrics, '
        'and switch the active query target with complete schema isolation.'
        '</p>'
        '</div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)

    datasets = fetch_datasets()
    active_id = st.session_state.get("active_dataset_id", "demo_lakehouse")

    # Ensure active_id exists in fetched datasets
    valid_ids = [d["dataset_id"] for d in datasets]
    if active_id not in valid_ids:
        active_id = "demo_lakehouse"
        st.session_state.active_dataset_id = active_id

    # Two-column layout: Left = Uploader & Catalog, Right = Data Profiling Report
    col_left, col_right = st.columns([1, 1.25], gap="large")

    with col_left:
        st.markdown(
            '<div style="font-size:0.75rem; font-family:var(--font-mono); color:var(--text-muted); '
            'text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">'
            '1. INGEST TABULAR DATASET</div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(
                '<div style="font-size:0.8125rem; color:var(--text-secondary); margin-bottom:0.75rem;">'
                'Supported Formats: <code>.csv</code>, <code>.parquet</code>, <code>.xlsx</code>, <code>.json</code> (Max: 5 GB Bounded Streaming)'
                '</div>',
                unsafe_allow_html=True,
            )

            uploaded_file = st.file_uploader(
                "Upload a tabular file",
                type=["csv", "xlsx", "xls", "parquet", "json", "jsonl"],
                label_visibility="collapsed",
                key="dataset_file_uploader",
            )

            if uploaded_file is not None:
                st.markdown(
                    f'<div style="font-size:0.75rem; font-family:var(--font-mono); color:var(--text-secondary); margin:0.5rem 0;">'
                    f'Selected: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)</div>',
                    unsafe_allow_html=True,
                )

                if st.button("Process & Register Dataset", type="primary", use_container_width=True):
                    with st.spinner("Ingesting, converting to Parquet, and profiling..."):
                        success, message, upload_result = upload_dataset_file(uploaded_file)
                        if success and upload_result:
                            st.success(message)
                            new_id = upload_result["metadata"]["dataset_id"]
                            st.session_state.active_dataset_id = new_id
                            st.rerun()
                        else:
                            st.error(message)

        st.markdown(
            '<div style="font-size:0.75rem; font-family:var(--font-mono); color:var(--text-muted); '
            'text-transform:uppercase; letter-spacing:0.05em; margin:1.5rem 0 0.5rem 0;">'
            '2. REGISTERED LAKEHOUSE DATASETS</div>',
            unsafe_allow_html=True,
        )

        for ds in datasets:
            ds_id = ds["dataset_id"]
            is_active = (ds_id == active_id)
            is_demo = ds.get("is_demo", False)
            border_color = "var(--accent-primary)" if is_active else "var(--border-subtle)"

            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    status_badge = (
                        '<span style="background:rgba(14,165,233,0.15); color:#38bdf8; border:1px solid rgba(14,165,233,0.3); '
                        'padding:0.125rem 0.375rem; border-radius:3px; font-size:0.625rem; font-family:var(--font-mono); font-weight:600;">ACTIVE TARGET</span>'
                        if is_active
                        else '<span style="color:var(--text-muted); font-size:0.625rem; font-family:var(--font-mono);">STANDBY</span>'
                    )
                    type_badge = (
                        '<span class="tier-badge tier-gold">DEMO LAKEHOUSE</span>'
                        if is_demo
                        else '<span class="tier-badge tier-silver">USER DATASET</span>'
                    )

                    card_text = (
                        f'<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.25rem;">'
                        f'{type_badge} {status_badge}'
                        f'</div>'
                        f'<div style="font-weight:600; font-size:0.875rem; color:var(--text-primary);">'
                        f'{html.escape(ds.get("name", ds_id))}'
                        f'</div>'
                        f'<div style="font-family:var(--font-mono); font-size:0.6875rem; color:var(--text-muted); margin-top:0.25rem;">'
                        f'Table: <code>{html.escape(ds.get("table_name", "orders"))}</code> &bull; '
                        f'Rows: {ds.get("row_count", 0):,} &bull; Cols: {ds.get("column_count", 0)}'
                        f'</div>'
                    )
                    st.markdown(card_text, unsafe_allow_html=True)

                with c2:
                    if not is_active:
                        if st.button("Set Active", key=f"activate_{ds_id}", use_container_width=True):
                            st.session_state.active_dataset_id = ds_id
                            st.rerun()
                    if not is_demo:
                        if st.button("Delete", key=f"del_{ds_id}", use_container_width=True):
                            ok, msg = delete_dataset(ds_id)
                            if ok:
                                if st.session_state.get("active_dataset_id") == ds_id:
                                    st.session_state.active_dataset_id = "demo_lakehouse"
                                st.rerun()
                            else:
                                st.error(msg)

        # Recent Async Jobs
        recent_jobs = fetch_jobs(limit=5)
        if recent_jobs:
            st.markdown(
                '<div style="font-size:0.75rem; font-family:var(--font-mono); color:var(--text-muted); '
                'text-transform:uppercase; letter-spacing:0.05em; margin:1.5rem 0 0.5rem 0;">'
                '3. RECENT INGESTION & GENERATION JOBS</div>',
                unsafe_allow_html=True,
            )
            for j in recent_jobs:
                j_stat = j.get("status", "unknown")
                badge_color = "#34d399" if j_stat == "completed" else ("#f87171" if j_stat == "failed" else "#38bdf8")
                with st.container(border=True):
                    st.markdown(
                        f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                        f'<span style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-primary); font-weight:600;">{j.get("job_id", "")}</span>'
                        f'<span style="font-family:var(--font-mono); font-size:0.625rem; color:{badge_color}; text-transform:uppercase; border:1px solid {badge_color}40; padding:1px 5px; border-radius:3px;">{j_stat}</span>'
                        f'</div>'
                        f'<div style="font-size:0.6875rem; color:var(--text-secondary); margin-top:0.25rem;">'
                        f'Type: <code>{j.get("job_type", "")}</code> &bull; Rows: {j.get("rows_processed", 0):,} &bull; Duration: {j.get("duration_sec", 0.0):.2f}s'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    with col_right:
        st.markdown(
            '<div style="font-size:0.75rem; font-family:var(--font-mono); color:var(--text-muted); '
            'text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">'
            'DATA QUALITY PROFILE & AUDIT</div>',
            unsafe_allow_html=True,
        )

        detail = fetch_dataset_detail(active_id)
        if not detail:
            st.info("Loading dataset audit details...")
            return

        meta = detail.get("metadata", {})
        profile = detail.get("profile", {})
        preview_rows = detail.get("preview", [])

        # Active Dataset Summary Card
        banner_summary_html = (
            f'<div class="dataset-meta-banner">'
            f'<div class="dataset-meta-item">'
            f'<span class="dataset-meta-label">Active Target</span>'
            f'<span class="dataset-meta-value">{html.escape(meta.get("name", "Demo"))}</span>'
            f'</div>'
            f'<div class="dataset-meta-item">'
            f'<span class="dataset-meta-label">Total Rows</span>'
            f'<span class="dataset-meta-value">{meta.get("row_count", 0):,}</span>'
            f'</div>'
            f'<div class="dataset-meta-item">'
            f'<span class="dataset-meta-label">Columns</span>'
            f'<span class="dataset-meta-value">{meta.get("column_count", 0)}</span>'
            f'</div>'
            f'<div class="dataset-meta-item">'
            f'<span class="dataset-meta-label">Duplicate Rows</span>'
            f'<span class="dataset-meta-value">{profile.get("duplicate_rows", 0):,} ({profile.get("duplicate_pct", 0.0)}%)</span>'
            f'</div>'
            f'<div class="dataset-meta-item">'
            f'<span class="dataset-meta-label">Missing Cells</span>'
            f'<span class="dataset-meta-value">{profile.get("missing_cells_pct", 0.0)}%</span>'
            f'</div>'
            f'</div>'
        )
        st.markdown(banner_summary_html, unsafe_allow_html=True)

        # Quality Warnings
        warnings = profile.get("warnings", [])
        if warnings:
            st.markdown(
                '<div style="font-size:0.6875rem; font-family:var(--font-mono); color:#fbbf24; '
                'text-transform:uppercase; margin:0.75rem 0 0.25rem 0;">Data Quality Alerts</div>',
                unsafe_allow_html=True,
            )
            for w in warnings:
                st.markdown(
                    f'<div style="background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.25); '
                    f'padding:0.375rem 0.625rem; border-radius:4px; font-size:0.75rem; color:#fde68a; margin-bottom:0.375rem;">'
                    f'&bull; {html.escape(w)}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25); '
                'padding:0.375rem 0.625rem; border-radius:4px; font-size:0.75rem; color:#a7f3d0; margin:0.5rem 0;">'
                'Zero data quality anomalies detected. Schema is ready for analytical queries.</div>',
                unsafe_allow_html=True,
            )

        # Column Health Breakdown
        columns = profile.get("columns", [])
        if columns:
            st.markdown(
                '<div style="font-size:0.6875rem; font-family:var(--font-mono); color:var(--text-muted); '
                'text-transform:uppercase; margin:1rem 0 0.5rem 0;">Column Schema & Distribution Health</div>',
                unsafe_allow_html=True,
            )

            col_summary_list = []
            for c in columns:
                type_badge = c.get("dtype", "VARCHAR")
                rag_flag = "TEXT (RAG Indexed)" if c.get("is_text") else ("NUMERIC" if c.get("is_numeric") else "CATEGORICAL")
                col_summary_list.append({
                    "Column Name": c.get("name"),
                    "SQL Type": type_badge,
                    "Nulls (%)": f"{c.get('null_pct', 0.0)}%",
                    "Distinct (%)": f"{c.get('unique_pct', 0.0)}%",
                    "Role": rag_flag,
                })

            col_df = pd.DataFrame(col_summary_list)
            st.dataframe(col_df, use_container_width=True, hide_index=True)

        # Sample Preview
        if preview_rows:
            st.markdown(
                '<div style="font-size:0.6875rem; font-family:var(--font-mono); color:var(--text-muted); '
                'text-transform:uppercase; margin:1rem 0 0.25rem 0;">Sample Data Records (First 50 Rows)</div>',
                unsafe_allow_html=True,
            )
            prev_df = pd.DataFrame(preview_rows)
            st.dataframe(prev_df, use_container_width=True, hide_index=True)
