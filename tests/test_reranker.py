import pytest
from src.schema import DocumentChunk, SearchResult
from src.reranker import Reranker


def test_reranker_basic():
    reranker = Reranker()

    chunk1 = DocumentChunk(id="c1", content="How to build state workshops to lower corruption in commanderies.")
    chunk2 = DocumentChunk(id="c2", content="Recruiting heavy cavalry retinues for flanking spear units.")
    chunk3 = DocumentChunk(id="c3", content="Managing public order and tax rates in major settlements.")

    initial_results = [
        SearchResult(chunk=chunk2, score=0.5, rank=1),
        SearchResult(chunk=chunk1, score=0.4, rank=2),
        SearchResult(chunk=chunk3, score=0.3, rank=3),
    ]

    query = "corruption reduction in commanderies"
    reranked = reranker.rerank(query, initial_results, top_k=2)

    assert len(reranked) == 2
    # Chunk 1 (corruption) should be re-ranked to position #1
    assert reranked[0].chunk.id == "c1"
    assert reranked[0].rank == 1
