import pytest
from src.schema import DocumentChunk
from src.qdrant_retriever import QdrantRetriever


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            id="doc_1",
            content="To lower corruption in your commandery, build a Grand Inspectorate.",
            metadata={"category": "economy"},
        ),
        DocumentChunk(
            id="doc_2",
            content="Cao Cao uses Credibility mechanics to incite proxy wars against enemy warlords.",
            metadata={"category": "diplomacy"},
        ),
        DocumentChunk(
            id="doc_3",
            content="Deploy spear infantry in mountain pass choke points to stop enemy cavalry charges.",
            metadata={"category": "military"},
        ),
    ]


def test_qdrant_retriever_in_memory(sample_chunks):
    # Test in-memory Qdrant instance
    retriever = QdrantRetriever(collection_name="test_collection", in_memory=True)
    retriever.index_chunks(sample_chunks)

    # Search for corruption management
    results = retriever.search("lower corruption in commandery", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.id == "doc_1"
    assert results[0].chunk.metadata.get("category") == "economy"
    assert results[0].score > 0.5
