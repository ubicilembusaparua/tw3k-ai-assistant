import re
from typing import List, Optional, Sequence
from rank_bm25 import BM25Okapi
from tw3k_ai_assistant.retrieval.dataset import load_dataset
from tw3k_ai_assistant.retrieval.schema import DocumentChunk, SearchResult

class BM25Retriever:
    """Lexical keyword-based retrieval using BM25Okapi."""

    def __init__(self, chunks: Optional[Sequence[DocumentChunk]] = None):
        self.chunks = list(chunks) if chunks is not None else load_dataset()
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
