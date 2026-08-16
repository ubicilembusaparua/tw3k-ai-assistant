import pytest
from src.schema import DocumentChunk, SearchResult
from _evaluation.evaluation import RagasEvaluator


def test_ragas_evaluator_format_dataset():
    evaluator = RagasEvaluator()

    chunk1 = DocumentChunk(id="c1", content="Grand Inspectorate lowers corruption in commanderies.")
    chunk2 = DocumentChunk(id="c2", content="Authority administrators boost public order.")

    sample = [
        {
            "question": "How to lower corruption?",
            "ground_truth": "Build a Grand Inspectorate.",
            "retrieved_results": [
                SearchResult(chunk=chunk1, score=0.9, rank=1),
                SearchResult(chunk=chunk2, score=0.7, rank=2),
            ],
        }
    ]

    dataset = evaluator.format_dataset(sample)
    assert len(dataset) == 1
    assert dataset[0]["question"] == "How to lower corruption?"
    assert dataset[0]["contexts"] == [
        "Grand Inspectorate lowers corruption in commanderies.",
        "Authority administrators boost public order.",
    ]
    assert dataset[0]["ground_truth"] == "Build a Grand Inspectorate."


def test_ragas_evaluator_offline_heuristic():
    evaluator = RagasEvaluator()

    chunk = DocumentChunk(id="c1", content="Build Grand Inspectorate and assign Authority administrators.")
    sample = [
        {
            "question": "How to reduce corruption?",
            "ground_truth": "Grand Inspectorate Authority administrators",
            "retrieved_results": [SearchResult(chunk=chunk, score=0.95, rank=1)],
        }
    ]

    result = evaluator.evaluate_retriever(sample, retriever_name="TestRetriever")
    assert result["retriever"] == "TestRetriever"
    assert "context_precision" in result["scores"]
    assert "context_recall" in result["scores"]
    assert result["scores"]["context_precision"] > 0.0
