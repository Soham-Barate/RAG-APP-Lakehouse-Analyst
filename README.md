# E-commerce RAG Lakehouse Analyst (`ecom-rag-lakehouse-analyst`)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B.svg?style=flat&logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?style=flat&logo=docker)](https://www.docker.com/)

## 🌐 Quick Access Links

| Application Service | Direct URL | Description |
| :--- | :--- | :--- |
| **Streamlit Web UI** | **[http://localhost:8501](http://localhost:8501)** | Full interactive Lakehouse Analyst analytics console, profiler, and table explorer |
| **FastAPI Swagger Docs** | **[http://localhost:8000/docs](http://localhost:8000/docs)** | OpenAPI / Swagger interactive documentation and schema explorer |
| **Backend Health Probe** | **[http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)** | Live system and infrastructure status JSON endpoint |

A production-grade, flagship-quality Hybrid RAG application (Text-to-SQL + Vector Search) built on top of a modern Iceberg Lakehouse architecture (simulated via DuckDB / Parquet for local execution). Supports both built-in synthetic medallion lakehouse data and **real user-uploaded datasets (CSV, XLSX, Parquet, JSON)** with automated data profiling and dynamic schema-isolated SQL generation.

This repository showcases advanced data engineering and AI engineering patterns: LangGraph agentic cycles, semantic caching, self-correcting retrieval loops, reciprocal rank fusion (RRF), Microsoft Presidio guardrails, RAGAS evaluations, dynamic user dataset ingestion, and a premium Streamlit interface.

---

## 🚀 Key Capabilities: Real User-Uploaded Data

1. **Multi-Format Ingestion**: Drag-and-drop upload for CSV, Excel (`.xlsx`, `.xls`), Parquet, and JSON / JSONL with magic-byte and schema validation.
2. **Automated Data Profiling**: Instant analysis of row counts, missing cell rates, duplicate records, inferred SQL types, and quality risk alerts (e.g. constant columns, extreme nulls).
3. **Dynamic Schema-Adaptive Text-to-SQL**: Translates natural language questions into safe DuckDB SQL adapting dynamically to any uploaded dataset's schema.
4. **Strict Dataset Isolation**: Uploaded datasets are assigned unique IDs and isolated storage paths (`data/uploads/<id>/`). Queries on Dataset A cannot access Dataset B or synthetic data.
5. **Dataset-Scoped RAG**: Automatically detects prose/text columns and indexes them into PostgreSQL pgvector with `dataset_id` metadata isolation.
6. **Executive Streamlit Experience**: 4 dedicated operational views: Analysis Console, Dataset Manager & Profiler, Lakehouse Table Explorer, and Cluster Telemetry.

---

## 🏗️ Architecture Overview

The system bridges unstructured text data (customer reviews, support tickets, product descriptions) and structured analytical data (orders, inventory, customers) using a unified **Agentic Router**.

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI["Streamlit Chat UI<br/>Chat History | Sources | Feedback"]
    end

    subgraph "API Layer"
        API["FastAPI Backend<br/>REST + SSE Streaming + Auth"]
        CACHE["Semantic Cache<br/>Redis + Vector Similarity + Filter Scope"]
    end

    subgraph "Agentic RAG Pipeline — LangGraph"
        ROUTER{"Query Router Node<br/>Intent & Schema Classification"}
        SQL_NODE["Safe Text-to-SQL Node<br/>DuckDB Parquet Lakehouse"]
        QR["Query Rewriter / Decomposer"]
        HYDE["HyDE Generator"]
        RET["Hybrid Retriever<br/>Vector + BM25 + Multi-Query RRF"]
        RERANK["Cross-Encoder Reranker"]
        COMPRESS["Contextual Compressor"]
        GRADE{"Relevance Grader"}
        GEN["Response Generator"]
        GUARD["Guardrails Node<br/>Toxicity | PII Scrubbing | Grounding Gate"]
    end

    subgraph "Data & Storage Layer"
        PG["PostgreSQL + pgvector<br/>HNSW Index + BM25 tsvector"]
        PARQUET["Parquet Lakehouse Store<br/>Orders, Products, Customers"]
        DUCK["DuckDB In-Memory OLAP<br/>Analytical SQL Engine"]
        REDIS["Redis<br/>Semantic Cache + LRU Eviction"]
    end

    subgraph "Observability"
        SMITH["LangSmith Tracing"]
        PROM["Prometheus Metrics"]
        LOG["Structured Logging<br/>JSON + Correlation IDs"]
    end

    UI --> API
    API --> CACHE
    CACHE -->|miss| ROUTER
    ROUTER -->|sql| SQL_NODE
    ROUTER -->|vector| QR
    ROUTER -->|hybrid| SQL_NODE
    SQL_NODE -->|sql route: generate/think| GEN
    SQL_NODE -->|hybrid route: continue to vector| QR
    QR --> HYDE
    HYDE --> RET
    RET --> RERANK
    RERANK --> COMPRESS
    COMPRESS --> GRADE
    GRADE -->|relevant| GEN
    GRADE -->|retry if irrelevant| QR
    GEN --> GUARD
    GUARD --> API
    SQL_NODE --> DUCK
    DUCK --> PARQUET
    RET --> PG
    CACHE --> REDIS
    API --> SMITH
    API --> PROM
    API --> LOG
```

---

## 📋 Feature Status Matrix

To provide complete transparency between architectural design and current code capabilities, each component is categorized below:

| Feature / Subsystem | Status | Details & Implementation Scope |
| :--- | :---: | :--- |
| **Safe Text-to-SQL Node** | ✅ **Implemented & Tested** | Schema-aware DuckDB executor querying Parquet/JSON lakehouse tables with AST/token allowlisting, strict read-only boundaries, and row limits. |
| **LangGraph Dynamic Branching** | ✅ **Implemented & Tested** | Conditional router branching: `sql` (direct to SQL), `vector` (semantic retrieval), and `hybrid` (combines structured SQL table with qualitative documents). |
| **Reciprocal Rank Fusion (RRF)** | ✅ **Implemented & Tested** | True RRF: $\operatorname{RRF}(d) = \sum_{r \in R} \frac{1}{k + \operatorname{rank}_r(d)}$ across independent query/HyDE lists with per-list deduplication and stable IDs. |
| **Parent-Child Chunking Isolation**| ✅ **Implemented & Tested** | Parents stored exclusively in `parent_documents` (unembedded); child chunks (256 tokens) stored in searchable `documents` table. |
| **Token-Aware Embedding Truncation**| ✅ **Implemented & Tested** | `tiktoken` encoding truncation (`<= 8000 tokens`) in `EmbeddingService` preventing provider token limit errors. |
| **Atomic Rate Limiter** | ✅ **Implemented & Tested** | Redis Lua script executes atomic token deductions without race conditions; configurable fail-open/fail-closed modes. |
| **Startup & Readiness Probes** | ✅ **Implemented & Tested** | `/api/v1/health` and `/api/v1/ready` probes with critical service fail-fast checks (`RuntimeError` in production). |
| **Lakehouse Catalog** | ⚙️ **Simulated Locally** | Tables (`products`, `orders`, `customers`) simulated via DuckDB querying local Parquet datasets. Distributed Apache Iceberg catalog on S3/MinIO is a planned deployment target. |
| **Semantic Cache** | 🔌 **Optional / Config-Dependent** | Redis-backed cosine similarity cache; gracefully bypassed if Redis is unavailable in development mode. |
| **Cross-Encoder Reranker** | 🔌 **Optional / Config-Dependent** | Cohere Rerank API with automatic fallback to retriever RRF scores if Cohere key is omitted. |
| **Alternative Vector Stores** | 🔌 **Optional / Config-Dependent** | PostgreSQL + pgvector is the default primary provider; Pinecone Serverless client implemented for cloud deployments. |
| **Distributed Query (Trino/Spark)** | 🔮 **Planned Roadmap** | Distributed catalog access across remote S3/MinIO buckets. |

---

## 🔥 Key Features

1. **Hybrid RAG Routing (SQL + Vector)**:
   * **Structured Analytics**: Automatically routes tabular queries (e.g., *"Show me the top 5 products by revenue last quarter"*) to a text-to-SQL compiler querying the Iceberg / DuckDB tables.
   * **Unstructured Semantics**: Routes unstructured queries (e.g., *"What do customers say about the battery life of headphones?"*) to pgvector/Pinecone vector searches.
   * **Hybrid Fusion**: Merges both structured results and unstructured documents for contextual insights.

2. **Self-Correcting LangGraph Loop**:
   * Uses **Query Rewriting** and **HyDE (Hypothetical Document Embeddings)** to maximize retrieval recall.
   * Employs an LLM-based **Relevance Grader**. If the retrieved content falls below the relevance threshold, the system automatically rewrites the query and retries (up to 3 times) before falling back.

3. **Production Vector Store Architecture**:
   * **Parent-Child Chunking**: Indexes small child chunks (256 tokens) for precise vector matching, but retrieves larger parent document chunks (2000 tokens) to preserve rich context.
   * **RRF (Reciprocal Rank Fusion)**: Combines Postgres GIN index BM25 full-text keyword ranking with pgvector cosine similarity.

4. **Enterprise Guardrails & Caching**:
   * **PII Redaction**: Pre-screens inputs and post-screens outputs using Microsoft Presidio to scrub emails, phone numbers, credit cards, and SSNs.
   * **Semantic Caching**: Employs a Redis cache that stores and retrieves previous queries based on vector cosine similarity (> 0.95), reducing LLM costs and latency.
   * **Toxicity & Hallucination Gates**: Filters unsafe content and validates that answers are grounded in context.

5. **Premium UI (Streamlit)**:
   * Professional, sleek dark-theme user interface with glassmorphism cards.
   * Inline expandable citations with relevance scores.
   * Interactive filters (category, date range, department).
   * Live message streaming and thumbs-up/down feedback.

---

## 📁 Project Structure

```
ecom-rag-lakehouse-analyst/
├── .env.example                    # Template environment variables
├── .gitignore                      # Python/data build ignores
├── Dockerfile                      # Multi-stage production build
├── docker-compose.yml              # PostgreSQL, Redis, backend, frontend stack
├── LICENSE                         # MIT License
├── Makefile                        # Dev convenience shortcuts
├── pyproject.toml                  # Linting, type-checking, and test configs
├── requirements.txt                # Pinned production requirements
│
├── config/
│   └── settings.py                 # Pydantic v2 configuration manager
│
├── backend/
│   ├── main.py                     # FastAPI app factory and lifespan
│   ├── dependencies.py             # FastAPI dependencies (DB, cache, LLM)
│   ├── api/
│   │   ├── routes/                 # chat, ingest, health, and feedback routes
│   │   └── middleware/             # Rate limiter, auth stub, and request logger
│   ├── core/
│   │   ├── schemas.py              # Pydantic validation schemas
│   │   └── exceptions.py           # Structured error responses
│   ├── rag/
│   │   ├── graph.py                # LangGraph StateGraph composition & dynamic routing
│   │   ├── state.py                # Pipeline state model
│   │   ├── nodes/                  # sql_node, retriever, grader, router, guardrails
│   │   └── cache/                  # Redis semantic cache with filter scoping & LRU
│   ├── vectorstore/
│   │   ├── base.py                 # Abstract vector store interface
│   │   ├── pgvector_store.py       # pgvector + hybrid search implementation
│   │   └── pinecone_store.py       # Pinecone Serverless client
│   └── services/
│       ├── llm_service.py          # Multi-provider client (OpenAI, Claude, Groq)
│       └── observability.py        # LangSmith tracing + Prometheus metrics
│
├── ingestion/
│   ├── pipeline.py                 # Data extraction, chunking, embedding, loading
│   ├── connectors/                 # Iceberg/DuckDB and text loaders
│   ├── chunking/                   # Parent-child chunking strategy
│   └── synthetic/                  # 50,000+ e-commerce records generator
│
├── frontend/
│   ├── app.py                      # Streamlit application entrypoint
│   ├── components/                 # Chat interface, sidebar, sources, feedback
│   └── styles/
│       └── custom.css              # Custom CSS styles
│
├── evaluation/
│   ├── test_queries.py             # Golden test set (50+ queries)
│   ├── ragas_eval.py               # RAGAS metrics runner
│   ├── custom_metrics.py           # Latency percentiles & cache analytics
│   └── benchmark.py                # Benchmark runner and reporter
│
└── tests/                          # Automated Pytest suite
```

---

## 🚀 Local Setup

### Option 1: Docker Compose (Recommended)

To start the entire stack (PostgreSQL with pgvector, Redis, FastAPI Backend, and Streamlit Frontend) in one command:

1. Clone the repository and navigate to the folder.
2. Create your `.env` file from the template:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` and add your `OPENAI_API_KEY` (or configure Ollama/Groq keys).
4. Build and start the services:
   ```bash
   docker-compose up --build
   ```
5. Run the sample ingestion script to load the data (in a new terminal):
   ```bash
   docker-compose exec backend python -m ingestion.pipeline
   ```
6. Open your browser:
   * **Streamlit UI**: [http://localhost:8501](http://localhost:8501)
   * **FastAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option 2: Bare Metal / Local Run

#### 1. Prerequisites
Ensure you have **Python 3.11+** installed. You will also need running instances of **PostgreSQL** (with `pgvector` extension) and **Redis** (optional, fallback to local dict caching if unavailable).

#### 2. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

#### 3. Setup Environment Variables
Copy `.env.example` to `.env` and edit the values:
```bash
cp .env.example .env
```
Key configuration items:
* `OPENAI_API_KEY`: Required for default embeddings (`text-embedding-3-large`) and LLM (`gpt-4o`).
* `DATABASE_URL`: Connection string (e.g., `postgresql+asyncpg://postgres:postgres@localhost:5432/ecom_rag`).
* `REDIS_URL`: Connection string (e.g., `redis://localhost:6379/0`).

#### 4. Run Ingestion (Generates & Indexes Synthetic Data)
```bash
python -m ingestion.pipeline
```
This generates ~50k rows of synthetic orders, products, customers, and tickets, writes them to local Parquet files / DuckDB tables (simulating Iceberg), chunks/embeds them, and loads them into PostgreSQL.

#### 5. Start Backend API
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 6. Start Streamlit Frontend
```bash
streamlit run frontend/app.py
```

---

## 📊 Evaluation & Benchmarking

The project contains a built-in benchmarking suite that runs a golden test set of **50+ curated queries** covering order lookups, sales trends, out-of-domain edge cases, and PII intrusion.

It collects RAGAS metrics (**Faithfulness, Answer Relevancy, Context Recall**) alongside performance metrics (**Latency percentiles, token usage, cache efficiency**).

### Run the Benchmark
Ensure the backend API is running, then execute:
```bash
python -m evaluation.benchmark
```
A complete JSON report and markdown summary will be generated under the `evaluation/reports/` folder, compiling scores like:
* **Faithfulness**: How grounded the answer is in retrieved context.
* **Answer Relevancy**: How well the answer addresses the question.
* **P95 Latency**: System response speed.

---

## 🛠️ Testing

A comprehensive automated test suite covers API endpoints, RAG nodes, guardrails, and ingestion.

Run the tests using `pytest`:
```bash
# Run unit and integration tests
pytest

# Run tests with coverage reporting
pytest --cov=backend --cov=ingestion --cov-report=term-missing
```

---

## 🔮 Future Roadmap

* **Trino Integration**: Connect the backend to a real, live multi-cluster Trino/Spark workspace querying AWS S3 Iceberg tables.
* **Graph Embeddings**: Integrate Neo4j to store customer-product relationship graphs and combine graph searches with vector embeddings.
* **Local Embedding Fallbacks**: Built-in support for embedding models running inside Ollama (`mxbai-embed-large`) or Sentence-Transformers (`all-MiniLM-L6-v2`).

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](file:///d:/RAG%20app/LICENSE) file for details.
