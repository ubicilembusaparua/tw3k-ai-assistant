from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config import Settings
from src.service import AssistantService, DuplicateSubmissionError, EmptyIndexError


class CountClient:
    def __init__(self, count: int) -> None:
        self.point_count = count

    def get_collections(self) -> object:
        return object()

    def count(self, **_: object) -> object:
        return SimpleNamespace(count=self.point_count)


class MissingCollectionClient(CountClient):
    def collection_exists(self, _: str) -> bool:
        return False

    def count(self, **_: object) -> object:
        raise AssertionError("count must not run for a missing collection")


def test_readiness_distinguishes_models_key_qdrant_and_index() -> None:
    service = AssistantService(
        Settings.from_env({"OPENAI_API_KEY": "test-key"}),
        client_factory=lambda _: CountClient(3),  # type: ignore[arg-type]
    )

    readiness = service.readiness()

    assert readiness["ready"] is False
    components = readiness["components"]
    assert components["qdrant"]["ready"] is True  # type: ignore[index]
    assert components["qdrant"]["index_count"] == 3  # type: ignore[index]
    assert components["openai"]["ready"] is True  # type: ignore[index]
    assert components["embedding_model"]["ready"] is False  # type: ignore[index]


def test_empty_index_is_reported_before_models_load() -> None:
    def forbidden(_: Settings) -> object:
        raise AssertionError("model must not load for an empty index")

    service = AssistantService(
        Settings.from_env({}),
        client_factory=lambda _: CountClient(0),  # type: ignore[arg-type]
        embedder_factory=forbidden,  # type: ignore[arg-type]
        reranker_factory=forbidden,  # type: ignore[arg-type]
    )

    with pytest.raises(EmptyIndexError):
        service.ask("food")


def test_missing_collection_is_an_empty_index_not_qdrant_failure() -> None:
    service = AssistantService(
        Settings.from_env({}),
        client_factory=lambda _: MissingCollectionClient(0),  # type: ignore[arg-type]
    )

    readiness = service.readiness()

    assert readiness["components"]["qdrant"] == {  # type: ignore[index]
        "ready": True,
        "index_count": 0,
        "error": None,
    }
    with pytest.raises(EmptyIndexError):
        service.ask("food")


def test_duplicate_question_is_rejected() -> None:
    service = AssistantService(Settings.from_env({}))
    service._active_questions.add("same question")

    with pytest.raises(DuplicateSubmissionError):
        service.ask("same   question")
