import re
from typing import List
from rank_bm25 import BM25Okapi
from src.schema import DocumentChunk, SearchResult
from src.dataset import load_dataset

class BM25Retriever:
    """Lexical keyword-based retrieval using BM25Okapi."""

    def __init__(self):
        self.chunks = load_dataset()
        # Tokenize corpus into lowercase word tokens
        self.corpus_tokens = [self._tokenize(chunk.content) for chunk in self.chunks]
        # Initialize BM25 search index
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def _tokenize(self, text: str) -> List[str]:
        """Simple lowercase word tokenizer."""
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieve top-k chunks matching the query using BM25 scores."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Calculate scores for all chunks in corpus
        scores = self.bm25.get_scores(query_tokens)

        # Get indices of top-k highest scoring chunks
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                SearchResult(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                    rank=rank,
                )
            )
        return results
