"""Qdrant collection compatibility checks for the transcript index."""

from __future__ import annotations

from typing import Any


class CollectionCompatibilityError(RuntimeError):
    """Raised when an existing collection cannot store configured embeddings."""


def assert_collection_compatible(
    collection_info: Any,
    *,
    expected_dimensions: int,
    expected_distance: str = "Cosine",
) -> None:
    """Reject vector layouts that could create a partially incompatible index."""

    try:
        vectors = collection_info.config.params.vectors
    except AttributeError as error:
        raise CollectionCompatibilityError(
            "Qdrant collection response does not contain vector configuration"
        ) from error

    if isinstance(vectors, dict):
        raise CollectionCompatibilityError(
            "existing collection uses named vectors; an unnamed vector is required"
        )

    size = getattr(vectors, "size", None)
    distance_value = getattr(vectors, "distance", None)
    distance = getattr(distance_value, "value", distance_value)

    mismatches: list[str] = []
    if size != expected_dimensions:
        mismatches.append(
            f"dimensions are {size!r}, expected {expected_dimensions}"
        )
    if str(distance).casefold() != expected_distance.casefold():
        mismatches.append(
            f"distance is {distance!r}, expected {expected_distance!r}"
        )
    if mismatches:
        raise CollectionCompatibilityError(
            "incompatible existing Qdrant collection: " + "; ".join(mismatches)
        )
