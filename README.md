# E-commerce RAG Lakehouse Analyst (`ecom-rag-lakehouse-analyst`)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B.svg?style=flat&logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?style=flat&logo=docker)](https://www.docker.com/)

A production-grade, flagship-quality Hybrid RAG application (Text-to-SQL + Vector Search) built on top of a modern Iceberg Lakehouse architecture (simulated via DuckDB / Parquet for local execution). 

This repository showcases advanced data engineering and AI engineering patterns: LangGraph agentic cycles, semantic caching, self-correcting retrieval loops, reciprocal rank fusion (RRF), Microsoft Presidio guardrails, RAGAS evaluations, and a premium Streamlit interface.

---

## 🏗️ Architecture Overview

The system bridges unstructured text data (customer reviews, support tickets, product descriptions) and structured analytical data (orders, inventory, customers) using a unified **Agentic Router**.

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI["Streamlit Chat UI<br/>Chat History | Sources | Feedback"]
    end

    subgraph "API Layer"
        API["FastAPI Backend<br/>REST + WebSocket + Auth"]
        CACHE["Semantic Cache<br/>Redis + Vector Similarity"]
    end

    subgraph "Agentic RAG Pipeline — LangGraph"
        ROUTER["Query Router Node"]
        QR["Query Rewriter / Decomposer"]
        HYDE["HyDE Generator"]
        RET["Hybrid Retriever<br/>Vector + BM25"]
        RERANK["Cross-Encoder Reranker"]
        COMPRESS["Contextual Compressor"]
        GRADE["Relevance Grader"]
        GEN["Response Generator"]
        GUARD["Guardrails Node<br/>Toxicity | PII | Relevance"]
    end

    subgraph "Data & Storage Layer"
        PG["PostgreSQL + pgvector<br/>HNSW Index + BM25"]
        PINE["Pinecone Serverless<br/>Hybrid + Metadata"]
        META["Metadata Store<br/>PostgreSQL Tables"]
        REDIS["Redis<br/>Cache + Sessions"]
    end

    subgraph "Data Ingestion Pipeline"
        ICE["Iceberg Lakehouse<br/>Bronze / Silver / Gold"]
        DUCK["DuckDB / Trino<br/>SQL Query Engine"]
        CHUNK["Chunking Engine<br/>Parent + Child Strategy"]
        EMB["Embedding Service<br/>text-embedding-3-large"]
    end

    subgraph "Observability"
        SMITH["LangSmith Tracing"]
        PROM["Prometheus Metrics"]
        LOG["Structured Logging<br/>JSON + Correlation IDs"]
    end

    UI --> API
    API --> CACHE
    CACHE -->|miss| ROUTER
    ROUTER --> QR
    QR --> HYDE
    HYDE --> RET
    RET --> RERANK
    RERANK --> COMPRESS
    COMPRESS --> GRADE
    GRADE -->|relevant| GEN
    GRADE -->|irrelevant| QR
    GEN --> GUARD
    GUARD --> API
    RET --> PG
    RET --> PINE
    ICE --> DUCK
    DUCK --> CHUNK
    CHUNK --> EMB
    EMB --> PG
    EMB --> PINE
    API --> SMITH
    API --> PROM
    API --> LOG
```

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
│   │   ├── graph.py                # LangGraph StateGraph composition
│   │   ├── state.py                # Pipeline state model
│   │   ├── nodes/                  # Individual agentic nodes (grader, router, etc.)
│   │   └── cache/                  # Redis semantic cache
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
