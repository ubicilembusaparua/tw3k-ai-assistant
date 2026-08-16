import pytest
from src.dataset import load_dataset


def test_load_dataset_file():
    chunks = load_dataset("tw3k_dataset.jsonl")

    assert len(chunks) == 2655
    first_chunk = chunks[0]
    assert first_chunk.id == "dX3rps0oAn4_c0000"
    assert "Serious Trivia" in first_chunk.content or "Total War" in first_chunk.content
    assert first_chunk.metadata.get("video_id") == "dX3rps0oAn4"
    assert first_chunk.metadata.get("channel") == "Serious Trivia"
