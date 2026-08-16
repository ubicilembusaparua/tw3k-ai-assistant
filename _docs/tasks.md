# RAG Application Implementation Tasks (LangChain Integration)

This document outlines the step-by-step task breakdown for building a production-grade RAG (Retrieval-Augmented Generation) application using LangChain, leveraging the existing custom retrieval pipeline (BM25, Qdrant Vector, Hybrid RRF, and Cross-Encoder Reranker).

---

## Task Breakdown

### Phase 1: Environment & Dependency Specifications
- [ ] **Task 1.1: Request & Add Dependencies in `pyproject.toml`**
  - Prompt user for approval to add `langchain`, `langchain-core`, `langchain-community`, and `langchain-openai` (or chosen provider) into `pyproject.toml`.
  - Run `uv sync` to update virtual environment and lockfile.
- [ ] **Task 1.2: Environment Configuration & Secret Management**
  - Update `.env` template with required keys (`OPENAI_API_KEY`, `OPENAI_MODEL_NAME`, `QDRANT_URL`, `QDRANT_API_KEY`).
  - Configure robust configuration loading (e.g. fallback defaults, validation).

---

### Phase 2: Custom LangChain Retriever Adapter (`src/langchain_retriever.py`)
- [ ] **Task 2.1: Implement Custom LangChain Retriever Class**
  - Create `Tw3kLangChainRetriever` subclassing `langchain_core.retrievers.BaseRetriever`.
  - Wrap existing `HybridRetriever` and optional `Reranker` pipeline.
  - Implement `_get_relevant_documents(query: str, *, run_manager=None) -> List[Document]`.
  - Map internal `SearchResult` / `DocumentChunk` data models into `langchain_core.documents.Document` with metadata (source ID, title, score, rank).
- [ ] **Task 2.2: Unit Tests for LangChain Retriever Adapter**
  - Create `tests/test_langchain_retriever.py`.
  - Test `_get_relevant_documents` returning expected LangChain `Document` objects.
  - Verify metadata preservation and error handling (empty queries, missing documents).

---

### Phase 3: LangChain LCEL RAG Chain (`src/rag_chain.py`)
- [ ] **Task 3.1: Define System & Context Prompt Templates**
  - Create structured prompt template (`ChatPromptTemplate`) instructing the LLM to strictly base answers on retrieved context.
  - Include strict formatting rules for source citations and fallback when context is insufficient.
- [ ] **Task 3.2: Implement LCEL Chain Construction (`build_rag_chain`)**
  - Construct runnable chain using LangChain Expression Language (LCEL):
    `retriever | format_docs | prompt | llm | StrOutputParser()`
  - Implement document formatting helper function `format_docs(docs: List[Document]) -> str`.
  - Return formatted response along with source document metadata.
- [ ] **Task 3.3: Multi-turn Conversation & History Support**
  - Support chat memory / history using `RunnableWithMessageHistory` or history-aware retriever chain (`create_history_aware_retriever`).
  - Handle query contextualization for multi-turn dialogues.

---

### Phase 4: Main Application Interface & CLI (`src/app.py` & `main.py`)
- [ ] **Task 4.1: Implement High-level RAG Application Class (`src/app.py`)**
  - Build `RAGApp` class orchestrating retriever selection (BM25, Vector, Hybrid, Hybrid+Rerank), LLM initialization, and chain invocation.
  - Provide `.answer(query: str, session_id: Optional[str] = None)` method returning answer string and source document list.
- [ ] **Task 4.2: Implement Interactive CLI Entry Point (`main.py`)**
  - Build CLI for interactive Q&A session with options to select retriever mode, top-k documents, and view source citations.

---

### Phase 5: Testing, Evaluation & Documentation
- [ ] **Task 5.1: Unit & Integration Test Suite (`tests/test_rag_chain.py`)**
  - Write test cases using `FakeListChatModel` or mock responses to run fast pytest suites without calling live external LLM APIs.
  - Verify chain output structure, context handling, and memory state retention.
- [ ] **Task 5.2: End-to-End RAGAS Evaluation Integration**
  - Extend `_evaluation/evaluate_ragas.py` to evaluate the generated answers against ground-truth datasets using Faithfulness and Answer Relevance metrics.
- [ ] **Task 5.3: Update Project Documentation (`README.md`)**
  - Update `README.md` with instructions on running the LangChain RAG application and CLI.

---

## Acceptance Criteria

1. **Clean Integration**: Seamless wrapping of existing `HybridRetriever` / `Reranker` into LangChain standard `BaseRetriever`.
2. **Deterministic Tests**: 100% of test suite in `tests/` passes via `uv run pytest` without requiring external API keys.
3. **Reproducible Environment**: Dependencies defined in `pyproject.toml` managed via `uv`.
4. **Git Compliance**: All changes cleanly committed in accordance with repository rules.
