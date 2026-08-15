from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.index_contract import (
    CollectionCompatibilityError,
    assert_collection_compatible,
)


def collection(*, size: int = 384, distance: object = "Cosine") -> object:
    vectors = SimpleNamespace(size=size, distance=distance)
    return SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=vectors))
    )


def test_compatible_collection_is_accepted() -> None:
    assert_collection_compatible(collection(), expected_dimensions=384)


@pytest.mark.parametrize(
    ("info", "message"),
    [
        (collection(size=768), "dimensions"),
        (collection(distance="Dot"), "distance"),
        (
            SimpleNamespace(
                config=SimpleNamespace(params=SimpleNamespace(vectors={"dense": object()}))
            ),
            "named vectors",
        ),
    ],
)
def test_incompatible_collection_is_rejected(info: object, message: str) -> None:
    with pytest.raises(CollectionCompatibilityError, match=message):
        assert_collection_compatible(info, expected_dimensions=384)
