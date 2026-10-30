@echo off
title RAG Lakehouse Analyst Launcher
echo =======================================================
echo   RAG Lakehouse Analyst - Dual-Path Intelligence
echo =======================================================
echo.

REM Activate virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo [1/2] Starting FastAPI Backend on http://localhost:8000 ...
start "RAG Backend (FastAPI)" cmd /k "title RAG Backend && python -m uvicorn backend.main:create_app --factory --host 0.0.0.0 --port 8000"

echo [2/2] Starting Streamlit Executive UI on http://localhost:8501 ...
start "RAG Frontend (Streamlit)" cmd /k "title RAG Frontend && streamlit run frontend/app.py --server.port 8501"

echo.
echo =======================================================
echo   Services Launched!
echo   - Streamlit Frontend: http://localhost:8501
echo   - FastAPI Docs:       http://localhost:8000/docs
echo =======================================================
echo.
echo Opening Streamlit Web UI in default browser...
timeout /t 3 /nobreak >nul
start http://localhost:8501

