import pytest
from src.schema import DocumentChunk
from src.bm25_retriever import BM25Retriever
from src.qdrant_retriever import QdrantRetriever
from src.hybrid_retriever import HybridRetriever


@pytest.fixture
def sample_corpus():
    return [
        DocumentChunk(
            id="chunk_1",
            content="Cao Cao mobilized his infantry units to establish defensive fortifications around the commandery.",
        ),
        DocumentChunk(
            id="chunk_2",
            content="To stabilize the local economy, agricultural tax rates were reduced and public granaries opened.",
        ),
        DocumentChunk(
            id="chunk_3",
            content="Guerilla tactics and night ambushes proved effective against invading cavalry formations.",
        ),
        DocumentChunk(
            id="chunk_4",
            content="Diplomatic alliances with neighboring warlords secured peace along the southern borders.",
        ),
    ]


def test_bm25_retriever(sample_corpus):
    bm25 = BM25Retriever(sample_corpus)
    results = bm25.search("granaries public economy", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.id == "chunk_2"
    assert results[0].rank == 1


def test_qdrant_retriever(sample_corpus):
    qdrant = QdrantRetriever(collection_name="test_retrievers_qdrant", in_memory=True)
    qdrant.index_chunks(sample_corpus)
    results = qdrant.search("infantry fortification troop defense", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.id == "chunk_1"


def test_hybrid_retriever(sample_corpus):
    bm25 = BM25Retriever(sample_corpus)
    qdrant = QdrantRetriever(collection_name="test_hybrid_qdrant", in_memory=True)
    qdrant.index_chunks(sample_corpus)
    hybrid = HybridRetriever(bm25, qdrant)

    results = hybrid.search("Cao Cao infantry military defense", top_k=3)

    assert len(results) == 3
    assert results[0].chunk.id == "chunk_1"
