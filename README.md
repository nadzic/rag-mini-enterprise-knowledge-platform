# RAG Mini Enterprise Knowledge Platform

Production-style backend for document-grounded Q&A using Retrieval-Augmented Generation (RAG), with a Streamlit frontend for ingestion and querying.  
This project ingests PDF documents, stores semantic vectors in Qdrant, and answers questions using only retrieved context.

## Why This Project

Enterprise teams need trustworthy answers from internal documentation, not generic LLM output.  
This codebase demonstrates a practical RAG workflow with:

- event-driven ingestion and query orchestration (`Inngest`)
- end-to-end observability with traces/spans (`Langfuse`)
- clean service boundaries for embeddings, chunking, and vector search
- typed contracts for pipeline steps (`Pydantic` models in `rag_types`)
- unit testing with mocked external dependencies (`pytest`)

## What It Does

### 1) Ingest PDF (`rag/ingest_pdf`)

- loads and chunks a PDF (`services/pdf_loader.py`)
- generates embeddings for each chunk (`services/embeddings.py`)
- upserts vectors and metadata into Qdrant (`services/vector_store.py`)

### 2) Query (`rag/query`)

- embeds the user question
- retrieves top-k relevant contexts from Qdrant
- sends grounded context to the LLM
- returns answer + sources + context count

## Tech Stack

- **Language:** Python 3.11
- **API Runtime:** FastAPI
- **Event Orchestration:** Inngest
- **Observability:** Langfuse
- **Vector DB:** Qdrant
- **LLM + Embeddings:** OpenAI API (`gpt-4o-mini`, `text-embedding-3-large`)
- **Document Processing:** `pypdf`, `llama-index`
- **Validation/Types:** Pydantic
- **Frontend/UI:** Streamlit
- **Testing:** Pytest, pytest-asyncio
- **Tooling:** uv

## Architecture (High-Level)

```text
PDF -> Chunk -> Embed -> Qdrant (upsert)
Question -> Embed -> Qdrant (search) -> LLM (grounded prompt) -> Answer + Sources
```

The app is served from `main.py`, which wires FastAPI + Inngest functions:

- `rag_ingest_pdf` in `inngest_functions/ingest_pdf.py`
- `rag_query_pdf_ai` in `inngest_functions/query_pdf.py`

## Project Structure

```text
inngest_functions/   # event-driven workflows (ingest + query)
services/            # core logic (embeddings, PDF chunking, vector store)
rag_types/           # exported Pydantic models
tests/               # unit tests with mocks/fakes
frontend/            # Streamlit UI (upload + ask)
main.py              # FastAPI app bootstrap
```

## Local Setup

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- Docker (recommended for local Qdrant)
- OpenAI API key

### 1) Install dependencies

```bash
uv sync --group dev
```

### 2) Configure environment

Create `.env` with:

```bash
OPENAI_API_KEY=your_key_here
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

(`OPEN_AI_KEY` and `OPENAI_KEY` are also supported as fallbacks.)
If Langfuse keys are not set, tracing is skipped and the app still runs normally.

## Observability (Langfuse)

Langfuse tracing is integrated in:

- `inngest_functions/query_pdf.py` for query spans and LLM generation tracing
- `inngest_functions/ingest_pdf.py` for ingestion pipeline spans
- `services/langfuse_client.py` for shared Langfuse client initialization and flushing

### 3) Start Qdrant locally

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4) Run the API

```bash
uv run uvicorn main:app --reload
```

### 5) Run Inngest Dev Server

In a separate terminal, start Inngest and point it to your local FastAPI handler:

```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

Local ingest/query dev flow uses 3 running processes:

1. Qdrant (`docker run -p 6333:6333 qdrant/qdrant`)
2. API (`uv run uvicorn main:app --reload`)
3. Inngest Dev Server (`npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery`)

### 6) Run the Streamlit UI

```bash
cd frontend
uv run streamlit run ./streamlit_app.py
```

## Testing

Run the unit tests:

```bash
uv run --group dev pytest
```

Current baseline includes tests for:

- `rag_types` model validation
- `embed_texts` behavior (mocked OpenAI client)
- `QdrantVectorStore` init/upsert/search behavior (fake Qdrant client)

## Engineering Decisions

- **Event-driven workflows:** Ingestion and query are explicit, observable functions.
- **Typed boundaries:** Pydantic models make step input/output contracts clear.
- **Deterministic IDs for chunks:** UUID5 (`source_id:index`) avoids duplicate collisions.
- **Dependency isolation in tests:** External services are mocked/faked for fast, reliable CI.

## Trade-offs / Next Improvements

- Add integration tests against a real local Qdrant container.
- Add auth + multi-tenant metadata filtering.
- Add a reranker to improve retrieval quality before answer generation.
- Add reranking and citation span extraction.
- Add prompt management/versioning in Langfuse for LLM prompts.
- Test different chunk sizes and chunk overlaps, then compare retrieval/answer quality.
- Test different embedding models and compare quality/cost/latency trade-offs.
- Add hybrid search (vector + BM25/keyword) and compare against pure vector retrieval.
- Add ingestion status tracking and retries dashboard.
- Improve the Streamlit UI/UX and add richer query history.

