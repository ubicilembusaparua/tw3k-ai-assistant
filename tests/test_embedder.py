import numpy as np
import pytest
from src.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


def test_embedder_dimension(embedder):
    assert embedder.get_embedding_dimension() == 384


def test_embedder_encode_single(embedder):
    text = "Cao Cao's credibility mechanics in Total War Three Kingdoms"
    vec = embedder.encode(text)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    # Vector should be normalized to unit length
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-4)


def test_embedder_encode_batch(embedder):
    texts = [
        "First test text passage.",
        "Second passage regarding military strategy.",
    ]
    vectors = embedder.encode_batch(texts)

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, 384)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)
