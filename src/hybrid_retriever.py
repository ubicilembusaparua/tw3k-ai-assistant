from typing import Dict, List
from src.bm25_retriever import BM25Retriever
from src.schema import DocumentChunk, SearchResult
from src.vector_retriever import VectorRetriever


class HybridRetriever:
    """Hybrid Retrieval combining BM25 and Vector search using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        rrf_k: int = 60,
    ):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5, fetch_k: int = 20) -> List[SearchResult]:
        """Fetch top candidates from both retrievers and fuse them using RRF."""
        # 1. Retrieve top candidates from both models
        bm25_results = self.bm25_retriever.search(query, top_k=fetch_k)
        vector_results = self.vector_retriever.search(query, top_k=fetch_k)

        # 2. Accumulate RRF scores using rank position
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}

        # Add BM25 ranks
        for item in bm25_results:
            cid = item.chunk.id
            chunk_map[cid] = item.chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + item.rank))

        # Add Vector ranks
        for item in vector_results:
            cid = item.chunk.id
            chunk_map[cid] = item.chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + item.rank))

        # 3. Sort chunks by descending fused RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        # 4. Return top-k fused SearchResult objects
        results = []
        for rank, cid in enumerate(sorted_cids, start=1):
            results.append(
                SearchResult(
                    chunk=chunk_map[cid],
                    score=rrf_scores[cid],
                    rank=rank,
                )
            )
        return results
