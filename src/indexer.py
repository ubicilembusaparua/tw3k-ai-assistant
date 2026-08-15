"""Idempotent local MiniLM-to-Qdrant transcript indexer."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from src.config import QdrantUnavailableError, Settings
from src.dataset import TranscriptRecord, iter_jsonl, point_id_for_chunk, require_valid_jsonl
from src.index_contract import assert_collection_compatible


class Embedder(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs: object) -> Any: ...


class IndexClient(Protocol):
    def get_collections(self) -> object: ...

    def collection_exists(self, collection_name: str) -> bool: ...

    def create_collection(self, collection_name: str, **kwargs: object) -> object: ...

    def get_collection(self, collection_name: str) -> object: ...

    def upsert(self, collection_name: str, **kwargs: object) -> object: ...


@dataclass(slots=True)
class IndexStats:
    validated: int = 0
    embedded: int = 0
    upserted: int = 0
    skipped: int = 0
    failed: int = 0

    @property
    def complete(self) -> bool:
        return self.failed == 0 and (
            self.upserted == self.validated or self.skipped == self.validated
        )


class IndexingIncompleteError(RuntimeError):
    def __init__(self, stats: IndexStats, message: str) -> None:
        self.stats = stats
        super().__init__(message)


def batched(records: Iterable[TranscriptRecord], batch_size: int) -> Iterator[list[TranscriptRecord]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    batch: list[TranscriptRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _default_client(settings: Settings) -> IndexClient:
    from qdrant_client import QdrantClient

    return QdrantClient(url=settings.qdrant_url, timeout=10)


def _default_embedder(settings: Settings) -> Embedder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def _ensure_qdrant_ready(client: IndexClient, settings: Settings) -> None:
    from qdrant_client.http import models

    try:
        client.get_collections()
        if not client.collection_exists(settings.qdrant_collection):
            client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=models.VectorParams(
                    size=settings.embedding_dimensions,
                    distance=models.Distance.COSINE,
                ),
            )
        collection = client.get_collection(settings.qdrant_collection)
    except Exception as error:
        if error.__class__.__name__ == "CollectionCompatibilityError":
            raise
        raise QdrantUnavailableError(
            f"Qdrant is unavailable at {settings.qdrant_url}: {error}"
        ) from error

    assert_collection_compatible(
        collection,
        expected_dimensions=settings.embedding_dimensions,
        expected_distance="Cosine",
    )


def _vectors_as_lists(encoded: Any, expected_count: int, dimensions: int) -> list[list[float]]:
    raw_vectors = encoded.tolist() if hasattr(encoded, "tolist") else encoded
    vectors = [list(vector) for vector in raw_vectors]
    if len(vectors) != expected_count:
        raise ValueError(
            f"embedder returned {len(vectors)} vectors for {expected_count} records"
        )
    for vector in vectors:
        if len(vector) != dimensions:
            raise ValueError(
                f"embedder returned {len(vector)} dimensions, expected {dimensions}"
            )
    return vectors


def index_jsonl(
    path: str | Path,
    settings: Settings,
    *,
    batch_size: int = 64,
    validate_only: bool = False,
    client_factory: Callable[[Settings], IndexClient] = _default_client,
    embedder_factory: Callable[[Settings], Embedder] = _default_embedder,
) -> IndexStats:
    """Preflight then stream, embed, and upsert a JSONL dataset in batches."""

    report = require_valid_jsonl(path)
    stats = IndexStats(validated=report.valid_records)
    if validate_only:
        stats.skipped = stats.validated
        return stats

    client = client_factory(settings)
    _ensure_qdrant_ready(client, settings)
    embedder = embedder_factory(settings)

    from qdrant_client.http import models

    for record_batch in batched(iter_jsonl(path), batch_size):
        try:
            encoded = embedder.encode(
                [record.embedding_input for record in record_batch],
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            vectors = _vectors_as_lists(
                encoded, len(record_batch), settings.embedding_dimensions
            )
            stats.embedded += len(record_batch)
            points = [
                models.PointStruct(
                    id=point_id_for_chunk(record.chunk_id),
                    vector=vector,
                    payload=record.qdrant_payload(
                        embedding_model=settings.embedding_model,
                        dimensions=settings.embedding_dimensions,
                    ),
                )
                for record, vector in zip(record_batch, vectors, strict=True)
            ]
            client.upsert(
                collection_name=settings.qdrant_collection,
                points=points,
                wait=True,
            )
            stats.upserted += len(points)
        except Exception as error:
            stats.failed += len(record_batch)
            raise IndexingIncompleteError(
                stats,
                f"indexing failed after {stats.upserted} upserts: {error}",
            ) from error

    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl_path", type=Path, help="transcript JSONL file to index")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate all records without Qdrant or model access",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        stats = index_jsonl(
            args.jsonl_path,
            Settings.from_env(),
            batch_size=args.batch_size,
            validate_only=args.validate_only,
        )
    except IndexingIncompleteError as error:
        print(json.dumps(asdict(error.stats), sort_keys=True), file=sys.stderr)
        print(str(error), file=sys.stderr)
        return 1
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2

    print(json.dumps(asdict(stats), sort_keys=True))
    return 0 if stats.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
