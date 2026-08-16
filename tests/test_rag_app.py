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
    assert "Video Title:" in context
    assert "Content: Battle of Red Cliffs strategy" in context
    assert "ThreeKingdomsHistory" not in context


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
    assert "Serious Trivia" not in context
    assert "05:12" not in context


def test_rag_base_search_with_reranker():
    mock_retriever = MagicMock()
    mock_reranker = MagicMock()

    candidate_results = [
        SearchResult(
            chunk=DocumentChunk(id="1", content="Passage 1", metadata={}),
            score=0.03,
            rank=1,
        ),
        SearchResult(
            chunk=DocumentChunk(id="2", content="Passage 2", metadata={}),
            score=0.02,
            rank=2,
        ),
    ]
    mock_retriever.search.return_value = candidate_results

    reranked_results = [candidate_results[1]]
    mock_reranker.rerank.return_value = reranked_results

    app = RAGBase(index=mock_retriever, reranker=mock_reranker)
    results = app.search("strategy", num_results=1, fetch_k=20)

    mock_retriever.search.assert_called_once_with("strategy", top_k=20)
    mock_reranker.rerank.assert_called_once_with("strategy", candidate_results, top_k=1)
    assert len(results) == 1
    assert results[0].chunk.id == "2"
