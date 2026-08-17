from datetime import datetime, timezone

from dashboard_utils import (
    daily_metrics,
    filter_records,
    judge_metrics,
    model_metrics,
    summarize_records,
)
from db_query import ConversationMetric


def make_record(
    record_id: int,
    model: str,
    day: int,
    *,
    score: int | None = None,
    relevance: str | None = None,
) -> ConversationMetric:
    return ConversationMetric(
        id=record_id,
        question=f"Question {record_id}",
        answer="Answer",
        model=model,
        instructions="Instructions",
        prompt="Prompt",
        prompt_tokens=100,
        completion_tokens=40,
        total_tokens=140,
        response_time=2.0,
        cost=0.001,
        timestamp=datetime(2026, 8, day, tzinfo=timezone.utc),
        judge_relevance=relevance,
        user_score=score,
    )


def test_filter_records_by_model_and_inclusive_date_range():
    records = [make_record(1, "model-a", 1), make_record(2, "model-b", 2)]

    filtered = filter_records(records, models={"model-b"}, date_range=(records[1].timestamp.date(), records[1].timestamp.date()))

    assert [record.id for record in filtered] == [2]


def test_summarize_records_calculates_cost_latency_tokens_and_feedback():
    records = [make_record(1, "model-a", 1, score=1), make_record(2, "model-a", 1, score=-1)]

    summary = summarize_records(records)

    assert summary.total_requests == 2
    assert summary.total_cost == 0.002
    assert summary.avg_response_time == 2.0
    assert summary.avg_total_tokens == 140
    assert summary.positive_feedback_rate == 50.0


def test_aggregations_group_records_for_dashboard_charts():
    records = [
        make_record(1, "model-a", 1, relevance="RELEVANT"),
        make_record(2, "model-a", 1, relevance="RELEVANT"),
        make_record(3, "model-b", 2, relevance="NON_RELEVANT"),
    ]

    assert [row["requests"] for row in daily_metrics(records)] == [2, 1]
    assert [row["model"] for row in model_metrics(records)] == ["model-a", "model-b"]
    assert [row["requests"] for row in judge_metrics(records)] == [2, 0, 1, 0]
