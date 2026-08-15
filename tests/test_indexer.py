from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import Settings
from src.dataset import point_id_for_chunk
from src.index_contract import CollectionCompatibilityError
from src.indexer import IndexingIncompleteError, index_jsonl


FIXTURE = Path(__file__).parent / "fixtures" / "transcripts.jsonl"


class FakeEmbedder:
    def __init__(self, events: list[str] | None = None) -> None:
        self.batch_sizes: list[int] = []
        self.events = events

    def encode(self, sentences: list[str], **_: object) -> list[list[float]]:
        if self.events is not None:
            self.events.append("embed")
        self.batch_sizes.append(len(sentences))
        return [[float(len(sentence))] + [0.0] * 383 for sentence in sentences]


class FakeQdrant:
    def __init__(self, events: list[str] | None = None) -> None:
        self.exists = False
        self.events = events
        self.points: dict[str, object] = {}
        self.create_count = 0
        self.size = 384
        self.distance = "Cosine"

    def get_collections(self) -> object:
        if self.events is not None:
            self.events.append("connect")
        return object()

    def collection_exists(self, _: str) -> bool:
        return self.exists

    def create_collection(self, collection_name: str, **kwargs: object) -> object:
        del collection_name
        self.exists = True
        self.create_count += 1
        config = kwargs["vectors_config"]
        self.size = config.size  # type: ignore[attr-defined]
        self.distance = config.distance  # type: ignore[attr-defined]
        return object()

    def get_collection(self, _: str) -> object:
        vectors = SimpleNamespace(size=self.size, distance=self.distance)
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=vectors))
        )

    def upsert(self, collection_name: str, **kwargs: object) -> object:
        del collection_name
        for point in kwargs["points"]:  # type: ignore[union-attr]
            self.points[str(point.id)] = point
        return object()


def write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def fixture_records() -> list[dict[str, object]]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


def test_validation_only_uses_no_qdrant_or_embedding() -> None:
    def forbidden(_: Settings) -> object:
        raise AssertionError("service factory must not be called")

    stats = index_jsonl(
        FIXTURE,
        Settings.from_env({}),
        validate_only=True,
        client_factory=forbidden,  # type: ignore[arg-type]
        embedder_factory=forbidden,  # type: ignore[arg-type]
    )

    assert stats.validated == 3
    assert stats.skipped == 3
    assert stats.complete


def test_connectivity_precedes_embedding_and_batches_are_bounded() -> None:
    events: list[str] = []
    client = FakeQdrant(events)
    embedder = FakeEmbedder(events)

    stats = index_jsonl(
        FIXTURE,
        Settings.from_env({}),
        batch_size=2,
        client_factory=lambda _: client,
        embedder_factory=lambda _: embedder,
    )

    assert events[0] == "connect"
    assert embedder.batch_sizes == [2, 1]
    assert (stats.validated, stats.embedded, stats.upserted, stats.failed) == (3, 3, 3, 0)
    assert client.create_count == 1


def test_rerun_updates_without_duplicates_and_addition_grows_count(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    records = fixture_records()
    write_records(path, records[:2])
    client = FakeQdrant()
    embedder = FakeEmbedder()
    kwargs = {
        "client_factory": lambda _: client,
        "embedder_factory": lambda _: embedder,
    }

    index_jsonl(path, Settings.from_env({}), **kwargs)
    assert len(client.points) == 2
    original_id = point_id_for_chunk(str(records[0]["chunk_id"]))
    original_vector = client.points[original_id].vector  # type: ignore[attr-defined]

    index_jsonl(path, Settings.from_env({}), **kwargs)
    assert len(client.points) == 2

    records[0]["text"] = "Edited guidance with a different vector length."
    write_records(path, records[:2])
    index_jsonl(path, Settings.from_env({}), **kwargs)
    assert len(client.points) == 2
    assert client.points[original_id].vector != original_vector  # type: ignore[attr-defined]
    assert client.points[original_id].payload["text"] == records[0]["text"]  # type: ignore[attr-defined]

    write_records(path, records)
    index_jsonl(path, Settings.from_env({}), **kwargs)
    assert len(client.points) == 3


def test_incompatible_collection_is_rejected_before_model_load() -> None:
    client = FakeQdrant()
    client.exists = True
    client.size = 768
    loaded = False

    def load_embedder(_: Settings) -> FakeEmbedder:
        nonlocal loaded
        loaded = True
        return FakeEmbedder()

    with pytest.raises(CollectionCompatibilityError):
        index_jsonl(
            FIXTURE,
            Settings.from_env({}),
            client_factory=lambda _: client,
            embedder_factory=load_embedder,
        )

    assert not loaded


def test_embedding_failure_reports_incomplete_counts() -> None:
    class BrokenEmbedder(FakeEmbedder):
        def encode(self, sentences: list[str], **_: object) -> list[list[float]]:
            return [[0.0]] * len(sentences)

    with pytest.raises(IndexingIncompleteError) as caught:
        index_jsonl(
            FIXTURE,
            Settings.from_env({}),
            batch_size=2,
            client_factory=lambda _: FakeQdrant(),
            embedder_factory=lambda _: BrokenEmbedder(),
        )

    assert caught.value.stats.failed == 2
    assert not caught.value.stats.complete
