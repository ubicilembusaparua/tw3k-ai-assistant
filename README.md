# RAG Retrieval Learning Suite

A modular Python framework to learn, implement, and evaluate three core RAG retrieval paradigms side-by-side:
1. **BM25 Lexical Search** (`rank_bm25`): Exact keyword matching algorithm.
2. **Dense Vector Search** (`sentence-transformers` + Cosine Similarity): Semantic vector space embedding search.
3. **Hybrid RRF Search** (Reciprocal Rank Fusion): Fusing lexical and semantic ranks for balanced retrieval.

## Setup & Running

```bash
uv sync
uv run pytest
uv run python evaluate.py
```

## Streamlit application

Run the Streamlit application to use the assistant or inspect captured latency,
token usage, cost, model usage, and user feedback. Use the
interface selector in the sidebar to switch between Chat and LLM metrics:

```bash
uv run streamlit run app.py
```

The dashboard reads the PostgreSQL conversation log configured through the
`POSTGRES_*` environment variables.
