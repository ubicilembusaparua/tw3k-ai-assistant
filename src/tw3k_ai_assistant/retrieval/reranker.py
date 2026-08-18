from typing import List, Optional
from sentence_transformers import CrossEncoder
from tw3k_ai_assistant.retrieval.schema import SearchResult


class Reranker:
    """Cross-Encoder Document Re-ranker using cross-encoder/ms-marco-MiniLM-L-6-v2."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: List[SearchResult], top_k: int = 5) -> List[SearchResult]:
        """Re-ranks candidate search results using Cross-Encoder joint attention scoring."""
        if not query.strip() or not results:
            return []

        pairs = [(query, res.chunk.content) for res in results]
        scores = self.model.predict(pairs)

        scored_results = []
        for res, score in zip(results, scores):
            scored_results.append((float(score), res.chunk))

        # Sort descending by re-ranker cross-attention score
        scored_results.sort(key=lambda x: x[0], reverse=True)

        reranked = []
        for rank, (score, chunk) in enumerate(scored_results[:top_k], start=1):
            reranked.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                )
            )

        return reranked
