"""Application-level orchestration for the TW3K retrieval pipeline.

The retrieval components live in :mod:`src`: BM25 and Qdrant are combined by
``HybridRetriever`` and the optional cross-encoder is provided by ``Reranker``.
This module keeps the LLM-facing prompt/context code in one place and exposes
the complete retrieval flow through :class:`RAGBase`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, List, Optional, Sequence

from src.bm25_retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever
from src.qdrant_retriever import QdrantRetriever
from src.reranker import Reranker
from src.schema import SearchResult


DEFAULT_FETCH_K = 20

GPT_54_MINI_PRICING_USD_PER_MILLION = {
    "input": 0.75,
    "cached_input": 0.075,
    "output": 4.50,
}


def calculate_rag_cost(
    response_json: Mapping[str, Any] | str,
    model: str = "gpt-5.4-mini",
) -> dict[str, Any]:
    """Calculate the OpenAI cost for one RAG response.

    ``response_json`` may be a parsed JSON object or a JSON string.  The
    function accepts Responses API usage keys (``input_tokens`` and
    ``output_tokens``) and Chat Completions aliases (``prompt_tokens`` and
    ``completion_tokens``).

    The returned costs are in USD.  If cached-input usage is present, cached
    input is charged at the lower cached-input rate; otherwise all input is
    charged at the standard input rate.
    """

    if isinstance(response_json, str):
        response_json = json.loads(response_json)
    if not isinstance(response_json, Mapping):
        raise TypeError("response_json must be a JSON object or JSON string.")

    if model != "gpt-5.4-mini" and not model.startswith("gpt-5.4-mini-"):
        raise ValueError(f"Unsupported pricing model: {model}")

    usage = response_json.get("usage", response_json)
    if not isinstance(usage, Mapping):
        raise ValueError("The JSON response must contain a usage object.")

    input_tokens = _read_token_count(usage, "input_tokens", "prompt_tokens")
    output_tokens = _read_token_count(usage, "output_tokens", "completion_tokens")
    cached_input_tokens = _read_cached_input_tokens(usage)

    if cached_input_tokens > input_tokens:
        raise ValueError("cached input tokens cannot exceed total input tokens.")

    uncached_input_tokens = input_tokens - cached_input_tokens
    input_cost_usd = (
        uncached_input_tokens * GPT_54_MINI_PRICING_USD_PER_MILLION["input"]
        + cached_input_tokens * GPT_54_MINI_PRICING_USD_PER_MILLION["cached_input"]
    ) / 1_000_000
    output_cost_usd = (
        output_tokens * GPT_54_MINI_PRICING_USD_PER_MILLION["output"]
    ) / 1_000_000

    return {
        "model": model,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
        "input_cost_usd": round(input_cost_usd, 10),
        "output_cost_usd": round(output_cost_usd, 10),
        "total_cost_usd": round(input_cost_usd + output_cost_usd, 10),
    }


def _read_token_count(usage: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if value is not None:
            try:
                token_count = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be an integer.") from exc
            if token_count < 0:
                raise ValueError(f"{key} cannot be negative.")
            return token_count
    raise ValueError(f"Missing token usage field; expected one of: {', '.join(keys)}")


def _read_cached_input_tokens(usage: Mapping[str, Any]) -> int:
    direct_value = usage.get("cached_input_tokens")
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details")
    if direct_value is None and isinstance(details, Mapping):
        direct_value = details.get("cached_tokens")
    if direct_value is None:
        return 0

    try:
        cached_tokens = int(direct_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("cached input tokens must be an integer.") from exc
    if cached_tokens < 0:
        raise ValueError("cached input tokens cannot be negative.")
    return cached_tokens


INSTRUCTIONS = """
You are an expert AI Assistant specialized in Total War: Three Kingdoms game mechanics, strategy, lore, campaigns, and guides. Your sole function is to answer user queries using exclusively the retrieved video transcript context provided below.

---

### Context Schema
The retrieved context will consist of transcript passages from strategy guides and lore videos with the following schema:
* **`Content`**: The transcript passage text containing advice, mechanics, strategy, or lore.
* **`video_title`**: Title of the video guide or playthrough.

---

### Execution Rules

1. **Strict Grounding:** Answer questions using **only** explicit information contained within the provided context (`Content`, `video_title`). Do not extrapolate, infer, or utilize external world knowledge.
2. **Rejection Criteria:**
   * If the user query is irrelevant to Total War: Three Kingdoms, warlords, battles, campaign strategy, or game lore, state explicitly: *"This query is outside the scope of the Total War: Three Kingdoms knowledge base."*
3. **No Hallucinations:** Never fabricate campaign strategies, character traits, faction mechanics, or game statistics not present in the context.
4. **Formatting:** Present responses clearly and concisely. When available, synthesize information across passages and cite video titles to assist the player.
""".strip()

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


class RAGBase:
    """Coordinate hybrid retrieval, optional reranking, and LLM prompting.

    ``index`` is expected to be a ``HybridRetriever`` from ``src``.  The
    ``bm25_retriever`` and ``qdrant_retriever`` arguments are a convenience for
    constructing that object in this class.  Keeping ``index`` supported also
    makes it possible to inject a fake retriever in unit tests.
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
    ) -> None:
        if index is not None and (bm25_retriever is not None or qdrant_retriever is not None):
            raise ValueError("Pass either index or both component retrievers, not both.")

        if index is None and (bm25_retriever is not None or qdrant_retriever is not None):
            if bm25_retriever is None or qdrant_retriever is None:
                raise ValueError("Both bm25_retriever and qdrant_retriever are required.")
            index = HybridRetriever(bm25_retriever, qdrant_retriever, rrf_k=rrf_k)

        self.index = index
        self.hybrid_retriever = index
        self.reranker = reranker
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    @classmethod
    def from_src(
        cls,
        *,
        rerank: bool = False,
        reranker: Optional[Reranker] = None,
        collection_name: str = "tw3k_transcripts",
        qdrant_url: str = "http://localhost:6333",
        rrf_k: int = 60,
        llm_client: Any = None,
        instructions: str = INSTRUCTIONS,
        prompt_template: str = PROMPT_TEMPLATE,
        model: str = "gpt-5.4-mini",
    ) -> "RAGBase":
        """Build the pipeline from the retrievers implemented in ``src``.

        The cross-encoder is loaded only when ``rerank=True`` or an explicit
        ``reranker`` is supplied because loading it downloads/initializes a
        model and is not needed for plain hybrid retrieval.
        """

        bm25 = BM25Retriever()
        qdrant = QdrantRetriever(
            collection_name=collection_name,
            url=qdrant_url,
        )
        hybrid = HybridRetriever(bm25, qdrant, rrf_k=rrf_k)

        if reranker is None and rerank:
            reranker = Reranker()

        return cls(
            index=hybrid,
            reranker=reranker,
            llm_client=llm_client,
            instructions=instructions,
            prompt_template=prompt_template,
            model=model,
        )

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
        """Apply the ``src.Reranker`` cross-encoder when one is configured."""

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
                    f"Video Title: {chunk.metadata.get('video_title', '')}",
                    f"Content: {chunk.content}",
                ]
                context_chunks.append("\n".join(chunk_lines))
                continue

            # Keep support for dictionary-shaped results used by older callers.
            if isinstance(result, dict):
                content = result.get("text") or result.get("content") or result.get("desc_1", "")
                video_title = result.get("video_title") or result.get("name", "")
                chunk_lines = [f"--- Document {idx} ---", f"Content: {content}"]
                chunk_lines.insert(1, f"Video Title: {video_title}")
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

        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]
        return self.llm_client.responses.create(
            model=self.model,
            input=input_messages,
        )

    def rag(self, query: str) -> Any:
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        return self.llm(prompt)
