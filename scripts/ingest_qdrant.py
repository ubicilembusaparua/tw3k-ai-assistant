"""Populate the configured Qdrant collection from the tracked transcript dataset."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from tw3k_ai_assistant.retrieval.dataset import load_dataset
from tw3k_ai_assistant.retrieval.qdrant import QdrantRetriever


DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION = "tw3k_transcripts"
DEFAULT_QDRANT_BATCH_SIZE = 64
DEFAULT_QDRANT_FORCE_REINDEX = False
DEFAULT_DATASET_PATH = "data/tw3k_dataset.jsonl"

TRUE_VALUES = frozenset({"1", "true", "t", "yes", "y", "on"})
FALSE_VALUES = frozenset({"0", "false", "f", "no", "n", "off"})


@dataclass(frozen=True)
class IngestionConfig:
    qdrant_url: str
    collection_name: str
    batch_size: int
    force_reindex: bool
    dataset_path: Path


def parse_bool(value: str | bool, *, name: str = "boolean") -> bool:
    """Parse a documented boolean environment value or raise a clear error."""

    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    accepted = ", ".join(sorted(TRUE_VALUES | FALSE_VALUES))
    raise ValueError(f"{name} must be one of: {accepted}; received {value!r}")


def parse_positive_int(value: str | int, *, name: str = "integer") -> int:
    """Parse a strictly positive decimal integer."""

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer; received {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer; received {value!r}")
    return parsed


def read_config(environ: Mapping[str, str] | None = None) -> IngestionConfig:
    values = environ if environ is not None else os.environ
    return IngestionConfig(
        qdrant_url=values.get("QDRANT_URL") or DEFAULT_QDRANT_URL,
        collection_name=values.get("QDRANT_COLLECTION") or DEFAULT_QDRANT_COLLECTION,
        batch_size=parse_positive_int(
            values.get("QDRANT_BATCH_SIZE", str(DEFAULT_QDRANT_BATCH_SIZE)),
            name="QDRANT_BATCH_SIZE",
        ),
        force_reindex=parse_bool(
            values.get("QDRANT_FORCE_REINDEX", str(DEFAULT_QDRANT_FORCE_REINDEX)),
            name="QDRANT_FORCE_REINDEX",
        ),
        dataset_path=Path(values.get("DATASET_PATH") or DEFAULT_DATASET_PATH),
    )


def run(*, environ: Mapping[str, str] | None = None) -> int:
    """Run one idempotent ingestion pass and return a process-style status."""

    config = read_config(environ)
    print(f"Qdrant URL: {config.qdrant_url}")
    print(f"Qdrant collection: {config.collection_name}")
    print(f"Batch size: {config.batch_size}")
    print(f"Force reindex: {str(config.force_reindex).lower()}")

    try:
        chunks = load_dataset(config.dataset_path)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to read index dataset at {config.dataset_path}: {exc}"
        ) from exc

    if not chunks:
        raise RuntimeError(
            f"Dataset {config.dataset_path} contains no indexable chunks; "
            "refusing to report a ready Qdrant index."
        )

    retriever = QdrantRetriever(
        collection_name=config.collection_name,
        url=config.qdrant_url,
    )
    existing_count = retriever.get_point_count()

    if existing_count > 0 and not config.force_reindex:
        print(
            f"Skipping indexing: collection {config.collection_name!r} already "
            f"contains {existing_count} points and force reindex is disabled."
        )
        return 0

    if config.force_reindex:
        print(
            f"FORCE REINDEX ENABLED: collection {config.collection_name!r} "
            "will be deleted and recreated."
        )
    else:
        print(
            f"Indexing {len(chunks)} chunks into empty collection "
            f"{config.collection_name!r}."
        )

    retriever.index_chunks(
        chunks,
        batch_size=config.batch_size,
        force=config.force_reindex,
    )
    resulting_count = retriever.get_point_count()
    print(f"Qdrant point count after ingestion: {resulting_count}")
    if resulting_count != len(chunks):
        raise RuntimeError(
            f"Qdrant index is incomplete: expected {len(chunks)} points but "
            f"found {resulting_count}."
        )
    return 0


def main() -> None:
    try:
        run()
    except Exception as exc:
        raise SystemExit(f"Qdrant ingestion failed: {exc}") from exc


if __name__ == "__main__":
    main()
