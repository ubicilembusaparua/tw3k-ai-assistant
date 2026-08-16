from unittest.mock import MagicMock
from rag_app import RAGBase
from src.schema import DocumentChunk, SearchResult


def test_rag_base_search_with_hybrid_retriever():
    mock_retriever = MagicMock()
    mock_search_results = [
        SearchResult(
            chunk=DocumentChunk(
                id="1",
                content="Cao Cao's forces marched north.",
                metadata={"video_title": "Battle of Guandu"},
            ),
            score=0.033,
            rank=1,
        )
    ]
    mock_retriever.search.return_value = mock_search_results

    app = RAGBase(index=mock_retriever)
    results = app.search("Cao Cao", num_results=5)

    mock_retriever.search.assert_called_once_with("Cao Cao", top_k=5)
    assert len(results) == 1
    assert results[0].chunk.content == "Cao Cao's forces marched north."


def test_rag_base_build_context_search_result():
    app = RAGBase(index=None)
    search_results = [
        SearchResult(
            chunk=DocumentChunk(
                id="1",
                content="Battle of Red Cliffs strategy",
                metadata={"channel": "ThreeKingdomsHistory"},
            ),
            score=0.05,
            rank=1,
        )
    ]
    context = app.build_context(search_results)
    assert "--- Document 1 ---" in context
    assert "Content: Battle of Red Cliffs strategy" in context
    assert "channel: ThreeKingdomsHistory" in context


def test_rag_base_build_context_dict():
    app = RAGBase(index=None)
    dict_results = [
        {
            "text": "Position archers on the high ground to maximize range and damage.",
            "video_title": "Total War Three Kingdoms Battle Guide",
            "channel": "Serious Trivia",
            "formatted_time": "05:12",
        }
    ]
    context = app.build_context(dict_results)
    assert "--- Document 1 ---" in context
    assert "Content: Position archers on the high ground to maximize range and damage." in context
    assert "Video Title: Total War Three Kingdoms Battle Guide" in context
    assert "Channel: Serious Trivia" in context
    assert "Timestamp: 05:12" in context
