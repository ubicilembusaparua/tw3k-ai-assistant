from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import Settings
from src.evaluation import (
    EvaluationProfile,
    EvaluationQuestion,
    evaluate_profile,
    load_questions,
    run_evaluation,
)
from src.retrieval import INSUFFICIENT_EVIDENCE, READY, Passage, RetrievalResult


def passage(video_id: str, text: str, link: str = "https://youtube.com/watch?v=x&t=1s") -> Passage:
    return Passage(
        chunk_ids=(f"{video_id}_c0000",),
        video_id=video_id,
        chunk_indexes=(0,),
        text=text,
        video_title="Guide",
        channel="Channel",
        start_time=1,
        end_time=2,
        formatted_time="00:01 - 00:02",
        timestamp_link=link,
        video_url="https://youtube.com/watch?v=x",
        retrieval_score=0.8,
        rerank_score=0.9,
    )


def questions() -> tuple[EvaluationQuestion, ...]:
    return (
        EvaluationQuestion("known", "economy", "food?", True, ("video",), ("food",)),
        EvaluationQuestion("unknown", "unsupported", "Neptune?", False, (), ()),
    )


def fake_retrieve(question: str, settings: Settings) -> RetrievalResult:
    del settings
    if question == "food?":
        return RetrievalResult(READY, question, (passage("video", "food advice"),), None, 1)
    return RetrievalResult(INSUFFICIENT_EVIDENCE, question, (), "irrelevant", 1)


def test_question_loader_validates_contract(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "q1",
                "category": "economy",
                "question": "food?",
                "supported": True,
                "expected_video_ids": ["video"],
                "expected_terms": ["food"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_questions(path)

    assert loaded[0].expected_video_ids == ("video",)


def test_evaluation_records_all_required_metrics() -> None:
    result = evaluate_profile(questions(), Settings.from_env({}), fake_retrieve)

    assert result["metrics"]["supported_hit_rate"] == 1.0
    assert result["metrics"]["unsupported_rejection_rate"] == 1.0
    assert result["metrics"]["citation_validity"] == 1.0
    assert result["metrics"]["groundedness"] == 1.0
    assert "latency_ms_p95" in result["metrics"]


def test_invalid_citation_is_measured_not_silently_accepted() -> None:
    def invalid(_: str, settings: Settings) -> RetrievalResult:
        del settings
        return RetrievalResult(
            READY, "food?", (passage("video", "food", "javascript:bad"),), None, 1
        )

    result = evaluate_profile(questions()[:1], Settings.from_env({}), invalid)

    assert result["metrics"]["citation_validity"] == 0.0
    assert result["records"][0]["citation_valid"] is False


def test_tuning_selects_quality_before_latency() -> None:
    profiles = (
        EvaluationProfile("rejects", 10, 6, 3000, 1.0, 0),
        EvaluationProfile("accepts", 20, 8, 4000, 0.35, 1),
    )

    def threshold_sensitive(question: str, settings: Settings) -> RetrievalResult:
        if question == "food?" and settings.relevance_threshold < 1:
            return fake_retrieve(question, settings)
        return RetrievalResult(INSUFFICIENT_EVIDENCE, question, (), "gated", 1)

    report = run_evaluation(
        questions(), Settings.from_env({}), threshold_sensitive, profiles
    )

    assert report["selected_profile"] == "accepts"
    assert report["recommended_settings"]["candidate_count"] == 20


def test_diagnostic_only_profile_cannot_become_default() -> None:
    diagnostic = EvaluationProfile("diagnostic", 10, 6, 3000, 0.35, 0, False)
    eligible = EvaluationProfile("eligible", 20, 8, 4000, 1.0, 1)

    report = run_evaluation(
        questions(), Settings.from_env({}), fake_retrieve, (diagnostic, eligible)
    )

    assert report["selected_profile"] == "eligible"
