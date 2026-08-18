"""Application-level orchestration for the TW3K retrieval pipeline.

The retrieval components live in :mod:`src`: BM25 and Qdrant are combined by
``HybridRetriever`` and the optional cross-encoder is provided by ``Reranker``.
This module keeps the LLM-facing prompt/context code in one place and exposes
the complete retrieval flow through :class:`RAGBase`.
"""

from __future__ import annotations

import os
from typing import Any, Iterator, List, Optional, Sequence

from tw3k_ai_assistant.rag.query_rewriter import QueryRewriter
from tw3k_ai_assistant.retrieval.bm25 import BM25Retriever
from tw3k_ai_assistant.retrieval.hybrid import HybridRetriever
from tw3k_ai_assistant.retrieval.qdrant import QdrantRetriever
from tw3k_ai_assistant.retrieval.reranker import Reranker
from tw3k_ai_assistant.retrieval.schema import SearchResult


DEFAULT_FETCH_K = 20


INSTRUCTIONS = """
You are an expert AI Assistant specialized in Total War: Three Kingdoms game mechanics, strategy, lore, campaigns, and guides. Your sole function is to answer user queries using exclusively the retrieved video transcript context provided below.

---

### Context Schema
The retrieved context will consist of transcript passages from strategy guides and lore videos.
Use only the passage content to answer the user.

---

### Execution Rules

1. **Strict Grounding:** Answer questions using **only** explicit information contained within the provided transcript content. Do not extrapolate, infer, or utilize external world knowledge.
2. **Rejection Criteria:**
   * If the user query is irrelevant to Total War: Three Kingdoms, warlords, battles, campaign strategy, or game lore, state explicitly: *"This query is outside the scope of the Total War: Three Kingdoms knowledge base."*
3. **No Hallucinations:** Never fabricate campaign strategies, character traits, faction mechanics, or game statistics not present in the context.
4. **Formatting:** Present responses clearly and concisely. Do not include video titles, source IDs, citations, URLs, or external links unless the user explicitly asks for them.
""".strip()

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


class RAGBase:
    """Build and coordinate the default TW3K retrieval pipeline.

    When no index or component retrievers are supplied, the BM25, Qdrant, and
    hybrid retrievers are constructed automatically. Set ``auto_build=False``
    for a lightweight instance used only for prompt/context helpers, or pass
    an index to inject a custom retriever.
    """

    def __init__(
        self,
        index: Optional[HybridRetriever] = None,
        reranker: Optional[Reranker] = None,
        llm_client: Any = None,
        instructions: str = INSTRUCTIONS,
        prompt_template: str = PROMPT_TEMPLATE,
        model: str = "gpt-5.4-mini",
        *,
        bm25_retriever: Optional[BM25Retriever] = None,
        qdrant_retriever: Optional[QdrantRetriever] = None,
        rrf_k: int = 60,
        query_rewriter: Optional[QueryRewriter] = None,
        collection_name: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        auto_build: bool = True,
        rerank: Optional[bool] = None,
        use_query_rewriter: bool = True,
    ) -> None:
        if index is not None and (bm25_retriever is not None or qdrant_retriever is not None):
            raise ValueError("Pass either index or both component retrievers, not both.")

        default_pipeline = False

        if index is None and (bm25_retriever is not None or qdrant_retriever is not None):
            if bm25_retriever is None or qdrant_retriever is None:
                raise ValueError("Both bm25_retriever and qdrant_retriever are required.")
            index = HybridRetriever(bm25_retriever, qdrant_retriever, rrf_k=rrf_k)

        if index is None and bm25_retriever is None and qdrant_retriever is None and auto_build:
            bm25_retriever = BM25Retriever()
            configured_collection = (
                collection_name
                or os.getenv("QDRANT_COLLECTION")
                or "tw3k_transcripts"
            )
            configured_url = (
                qdrant_url
                or os.getenv("QDRANT_URL")
                or "http://localhost:6333"
            )
            qdrant_retriever = QdrantRetriever(
                collection_name=configured_collection,
                url=configured_url,
            )
            index = HybridRetriever(bm25_retriever, qdrant_retriever, rrf_k=rrf_k)
            default_pipeline = True

        if rerank is None:
            rerank = default_pipeline

        if reranker is None and rerank:
            reranker = Reranker()

        if query_rewriter is None and use_query_rewriter and llm_client is not None:
            query_rewriter = QueryRewriter(llm_client)

        self.index = index
        self.hybrid_retriever = index
        self.reranker = reranker
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model
        self.query_rewriter = query_rewriter

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = DEFAULT_FETCH_K,
    ) -> List[SearchResult]:
        """Run BM25 + Qdrant retrieval through ``HybridRetriever``.

        ``top_k`` is the number of fused results returned. ``fetch_k`` is the
        number of candidates requested from each underlying retriever before
        reciprocal-rank fusion.  The source implementation defaults to 20;
        omitting that argument for the default preserves compatibility with
        simple injected test doubles.
        """

        if self.hybrid_retriever is None or not query.strip() or top_k <= 0:
            return []

        if fetch_k == DEFAULT_FETCH_K:
            return self.hybrid_retriever.search(query, top_k=top_k)
        return self.hybrid_retriever.search(query, top_k=top_k, fetch_k=fetch_k)

    def rerank_results(
        self,
        query: str,
        results: Sequence[SearchResult],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Apply the retrieval reranker when one is configured."""

        if top_k <= 0 or not results:
            return []
        if self.reranker is None:
            return list(results[:top_k])
        return self.reranker.rerank(query, list(results), top_k=top_k)

    def search(
        self,
        query: str,
        num_results: int = 5,
        fetch_k: int = DEFAULT_FETCH_K,
    ) -> List[SearchResult]:
        """Retrieve final context, reranking hybrid candidates when enabled."""

        if num_results <= 0 or not query.strip():
            return []

        candidate_k = fetch_k if self.reranker is not None else num_results
        candidates = self.hybrid_search(
            query,
            top_k=candidate_k,
            fetch_k=fetch_k,
        )

        if self.reranker is None:
            return candidates
        return self.rerank_results(query, candidates, top_k=num_results)

    def build_context(self, search_results: Sequence[Any]) -> str:
        """Format source ``SearchResult`` objects for the grounded prompt."""

        context_chunks = []

        for idx, result in enumerate(search_results, start=1):
            if isinstance(result, SearchResult):
                chunk = result.chunk
                chunk_lines = [
                    f"--- Document {idx} ---",
                    f"Content: {chunk.content}",
                ]
                context_chunks.append("\n".join(chunk_lines))
                continue

            # Keep support for dictionary-shaped results used by older callers.
            if isinstance(result, dict):
                content = result.get("text") or result.get("content") or result.get("desc_1", "")
                chunk_lines = [f"--- Document {idx} ---", f"Content: {content}"]
                context_chunks.append("\n".join(chunk_lines))
                continue

            context_chunks.append(f"--- Document {idx} ---\n{result}")

        return "\n\n".join(context_chunks)

    def build_prompt(self, query: str, search_results: Sequence[Any]) -> str:
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt: str) -> Any:
        if self.llm_client is None:
            raise RuntimeError("An llm_client is required to generate an answer.")

        return self.llm_client.responses.create(
            model=self.model,
            input=self._build_llm_input(prompt),
        )

    def llm_stream(self, prompt: str) -> Iterator[str]:
        """Yield answer text deltas from the Responses API stream."""

        if self.llm_client is None:
            raise RuntimeError("An llm_client is required to generate an answer.")

        with self.llm_client.responses.stream(
            model=self.model,
            input=self._build_llm_input(prompt),
        ) as stream:
            for event in stream:
                if getattr(event, "type", None) != "response.output_text.delta":
                    continue
                delta = getattr(event, "delta", "")
                if delta:
                    yield delta

    def _build_llm_input(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]

    def rag(self, query: str) -> Any:
        retrieval_query = query
        if self.query_rewriter is not None:
            retrieval_query = self.query_rewriter.rewrite(query)

        search_results = self.search(retrieval_query)

        prompt = self.build_prompt(query, search_results)
        return self.llm(prompt)

    def rag_stream(self, query: str) -> Iterator[str]:
        """Run retrieval and stream the generated answer text."""

        retrieval_query = query
        if self.query_rewriter is not None:
            retrieval_query = self.query_rewriter.rewrite(query)

        search_results = self.search(retrieval_query)
        prompt = self.build_prompt(query, search_results)
        yield from self.llm_stream(prompt)
