from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from src.schema import DocumentChunk, SearchResult


class VectorRetriever:
    """Dense semantic vector retrieval using SentenceTransformers + Cosine Similarity."""

    def __init__(
        self,
        chunks: List[DocumentChunk],
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        
        # Pre-embed all corpus chunks into normalized 384-dim vectors
        texts = [chunk.content for chunk in chunks]
        self.chunk_embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieve top-k chunks using vector cosine similarity."""
        if not query.strip():
            return []

        # Embed query into normalized vector
        query_embedding = self.model.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        )

        # Dot product of normalized vectors equals cosine similarity scores
        scores = np.dot(self.chunk_embeddings, query_embedding)

        # Get indices of top-k highest scoring chunks
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                SearchResult(
                    chunk=self.chunks[int(idx)],
                    score=float(scores[idx]),
                    rank=rank,
                )
            )
        return results
