from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import QdrantUnavailableError, Settings
from src.retrieval import INSUFFICIENT_EVIDENCE, READY, Passage, retrieve, select_context


FIXTURE = Path(__file__).parent / "fixtures" / "transcripts.jsonl"


def records() -> list[dict[str, object]]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


class FakeEmbedder:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def encode(self, sentences: list[str], **kwargs: object) -> list[list[float]]:
        self.questions.extend(sentences)
        assert kwargs["normalize_embeddings"] is True
        return [[0.0] * 384 for _ in sentences]


class FakeReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]], **_: object) -> list[float]:
        self.pairs = pairs
        return self.scores


class FakeClient:
    def __init__(self, payloads: list[dict[str, object]], scores: list[float]) -> None:
        self.payloads = payloads
        self.scores = scores
        self.calls: list[dict[str, object]] = []

    def query_points(self, collection_name: str, **kwargs: object) -> object:
        self.calls.append({"collection_name": collection_name, **kwargs})
        points = [
            SimpleNamespace(payload=payload, score=score)
            for payload, score in zip(self.payloads, self.scores, strict=True)
        ]
        return SimpleNamespace(points=points)


def run_retrieval(
    payloads: list[dict[str, object]],
    rerank_scores: list[float],
    *,
    env: dict[str, str] | None = None,
) -> tuple[object, FakeClient, FakeEmbedder, FakeReranker]:
    client = FakeClient(payloads, [0.9 - index / 10 for index in range(len(payloads))])
    embedder = FakeEmbedder()
    reranker = FakeReranker(rerank_scores)
    result = retrieve(
        "How should I manage food?",
        Settings.from_env(env or {}),
        client_factory=lambda _: client,
        embedder_factory=lambda _: embedder,
        reranker_factory=lambda _: reranker,
    )
    return result, client, embedder, reranker


def test_retrieval_uses_configured_candidate_count_and_preserves_scores() -> None:
    result, client, embedder, reranker = run_retrieval(records(), [0.95, 0.85, 0.5])

    assert result.status == READY
    assert embedder.questions == ["How should I manage food?"]
    assert client.calls[0]["limit"] == 10
    assert client.calls[0]["with_payload"] is True
    assert reranker.pairs[0][1].startswith("Economy basics\nBuild food")
    assert result.passages[0].retrieval_score == pytest.approx(0.9)
    assert result.passages[0].rerank_score == pytest.approx(0.95)


def test_adjacent_overlap_is_joined_and_duplicate_does_not_dominate() -> None:
    payloads = records()
    payloads.append(dict(payloads[0], chunk_id="duplicate_c0000"))

    result, *_ = run_retrieval(payloads, [0.95, 0.9, 0.8, 0.85])

    assert result.status == READY
    assert len(result.passages) == 2
    joined = next(passage for passage in result.passages if passage.video_id == "video-a")
    assert joined.chunk_indexes == (0, 1)
    assert joined.timestamp_link.endswith("&t=1s")
    assert "Trade surplus food" in joined.text


def test_context_selection_honors_count_and_budget() -> None:
    settings = Settings.from_env(
        {"FINAL_CONTEXT_COUNT": "2", "CONTEXT_TOKEN_BUDGET": "20"}
    )
    candidates = [
        Passage(
            chunk_ids=(str(index),),
            video_id=str(index),
            chunk_indexes=(0,),
            text="x" * 40,
            video_title="title",
            channel="channel",
            start_time=0,
            end_time=1,
            formatted_time="00:00 - 00:01",
            timestamp_link=f"https://youtube.com/watch?v={index}&t=0s",
            video_url=f"https://youtube.com/watch?v={index}",
            retrieval_score=1.0,
            rerank_score=1.0 - index / 10,
        )
        for index in range(4)
    ]

    selected = select_context(candidates, settings)

    assert len(selected) == 2
    assert sum(passage.estimated_tokens for passage in selected) <= 20


def test_empty_or_irrelevant_results_are_deterministic() -> None:
    empty, *_ = run_retrieval([], [])
    irrelevant, *_ = run_retrieval(records(), [0.1, 0.2, 0.3])

    assert empty.status == INSUFFICIENT_EVIDENCE
    assert empty.passages == ()
    assert irrelevant.status == INSUFFICIENT_EVIDENCE
    assert "below relevance threshold" in irrelevant.reason


def test_invalid_source_link_is_never_selected() -> None:
    payload = records()[0]
    payload["timestamp_link"] = "javascript:alert(1)"

    result, *_ = run_retrieval([payload], [0.99])

    assert result.status == INSUFFICIENT_EVIDENCE
    assert result.passages == ()


def test_qdrant_error_is_not_converted_to_no_evidence() -> None:
    class BrokenClient:
        def query_points(self, collection_name: str, **kwargs: object) -> object:
            raise ConnectionError("Qdrant stopped")

    with pytest.raises(QdrantUnavailableError, match="Qdrant stopped"):
        retrieve(
            "food",
            Settings.from_env({}),
            client_factory=lambda _: BrokenClient(),
            embedder_factory=lambda _: FakeEmbedder(),
            reranker_factory=lambda _: FakeReranker([]),
        )
