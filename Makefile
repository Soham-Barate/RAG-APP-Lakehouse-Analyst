# ============================================================
# Makefile — Developer convenience commands
# ============================================================

.PHONY: help install dev run-backend run-frontend docker-up docker-down test lint generate-data ingest benchmark clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

dev: ## Run backend in development mode
	python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

run-backend: ## Run the FastAPI backend
	python -m backend.main

run-frontend: ## Run the Streamlit frontend
	python -m streamlit run frontend/app.py --server.port 8501

docker-up: ## Start all services via Docker Compose
	docker-compose up -d --build

docker-down: ## Stop all Docker services
	docker-compose down

docker-logs: ## Tail all Docker logs
	docker-compose logs -f

test: ## Run the test suite
	python -m pytest tests/ -v --cov=backend --cov=ingestion --cov-report=term-missing

lint: ## Run linting
	python -m ruff check backend/ ingestion/ evaluation/
	python -m ruff format --check backend/ ingestion/ evaluation/

format: ## Auto-format code
	python -m ruff format backend/ ingestion/ evaluation/

typecheck: ## Run type checking
	python -m mypy backend/ ingestion/ --ignore-missing-imports

generate-data: ## Generate synthetic demo data
	python -c "from ingestion.synthetic.data_generator import SyntheticDataGenerator; SyntheticDataGenerator().generate_all(); print('Done!')"

benchmark: ## Run the evaluation benchmark
	python -m evaluation.benchmark --quick

benchmark-full: ## Run full evaluation benchmark
	python -m evaluation.benchmark

clean: ## Clean temporary files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov dist build *.egg-info
