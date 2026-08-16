import pytest
from src.dataset import load_dataset


def test_load_dataset_file():
    chunks = load_dataset("tw3k_dataset.jsonl")

    assert len(chunks) == 1152
    first_chunk = chunks[0]
    assert first_chunk.id == "6P4APnVRX-8_c0000"
    assert "Serious Trivia" in first_chunk.content or "Hello everyone" in first_chunk.content
    assert first_chunk.metadata.get("video_id") == "6P4APnVRX-8"
