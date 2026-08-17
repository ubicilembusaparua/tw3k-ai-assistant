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

## LLM metrics dashboard

Run the Streamlit dashboard to inspect captured latency, token usage, cost,
model usage, judge relevance, and user feedback:

```bash
uv run streamlit run dashboard.py
```

The dashboard reads the PostgreSQL conversation log configured through the
`POSTGRES_*` environment variables.
