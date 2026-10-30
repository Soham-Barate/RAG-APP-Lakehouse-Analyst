# ============================================================
# Multi-stage Docker build for E-commerce RAG Assistant
# ============================================================

# Stage 1: Dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim as runtime

# Security: run as non-root user
RUN groupadd -r raguser && useradd -r -g raguser raguser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY config/ config/
COPY backend/ backend/
COPY ingestion/ ingestion/
COPY evaluation/ evaluation/
COPY frontend/ frontend/

# Create data directory
RUN mkdir -p data logs && chown -R raguser:raguser /app

USER raguser

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/api/v1/health'); exit(0 if r.status_code == 200 else 1)" || exit 1

# Default: run the FastAPI backend
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
