"""Pure data transformations used by the Streamlit metrics dashboard."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from typing import Collection, Optional, Sequence

from db_query import ConversationMetric


@dataclass(frozen=True)
class DashboardSummary:
    total_requests: int
    total_cost: float
    avg_response_time: float
    avg_total_tokens: float
    avg_prompt_tokens: float
    avg_completion_tokens: float
    rated_feedback: int
    positive_feedback_rate: Optional[float]


def _record_date(record: ConversationMetric) -> date:
    if isinstance(record.timestamp, datetime):
        return record.timestamp.date()
    return record.timestamp


def filter_records(
    records: Sequence[ConversationMetric],
    models: Optional[Collection[str]] = None,
    date_range: Optional[tuple[date, date]] = None,
) -> list[ConversationMetric]:
    """Filter records by model and inclusive calendar date range."""

    model_set = set(models) if models is not None else None
    start_date, end_date = date_range or (None, None)

    return [
        record
        for record in records
        if (model_set is None or record.model in model_set)
        and (start_date is None or _record_date(record) >= start_date)
        and (end_date is None or _record_date(record) <= end_date)
    ]


def summarize_records(records: Sequence[ConversationMetric]) -> DashboardSummary:
    """Calculate headline metrics from the currently filtered records."""

    total = len(records)
    if total == 0:
        return DashboardSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, None)

    user_scores = [record.user_score for record in records if record.user_score in (-1, 1)]
    return DashboardSummary(
        total_requests=total,
        total_cost=sum(record.cost for record in records),
        avg_response_time=sum(record.response_time for record in records) / total,
        avg_total_tokens=sum(record.total_tokens for record in records) / total,
        avg_prompt_tokens=sum(record.prompt_tokens for record in records) / total,
        avg_completion_tokens=sum(record.completion_tokens for record in records) / total,
        rated_feedback=len(user_scores),
        positive_feedback_rate=(
            sum(score == 1 for score in user_scores) / len(user_scores) * 100
            if user_scores
            else None
        ),
    )


def daily_metrics(records: Sequence[ConversationMetric]) -> list[dict[str, object]]:
    """Aggregate request count, cost, latency, and tokens by calendar day."""

    groups: dict[str, list[ConversationMetric]] = defaultdict(list)
    for record in records:
        groups[_record_date(record).isoformat()].append(record)

    rows = []
    for day, day_records in sorted(groups.items()):
        count = len(day_records)
        rows.append(
            {
                "date": day,
                "requests": count,
                "cost": sum(record.cost for record in day_records),
                "avg_response_time": sum(record.response_time for record in day_records) / count,
                "avg_tokens": sum(record.total_tokens for record in day_records) / count,
            }
        )
    return rows


def model_metrics(records: Sequence[ConversationMetric]) -> list[dict[str, object]]:
    """Aggregate request count, cost, and latency by model."""

    groups: dict[str, list[ConversationMetric]] = defaultdict(list)
    for record in records:
        groups[record.model].append(record)

    rows = []
    for model, model_records in sorted(groups.items()):
        count = len(model_records)
        rows.append(
            {
                "model": model,
                "requests": count,
                "cost": sum(record.cost for record in model_records),
                "avg_response_time": sum(record.response_time for record in model_records) / count,
            }
        )
    return rows


def recent_rows(records: Sequence[ConversationMetric]) -> list[dict[str, object]]:
    """Create compact, user-facing rows for the recent requests table."""

    rows = []
    for record in records:
        rows.append(
            {
                "time": record.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "question": record.question,
                "model": record.model,
                "response_time_s": round(record.response_time, 3),
                "total_tokens": record.total_tokens,
                "cost_usd": round(record.cost, 6),
                "user_score": record.user_score,
            }
        )
    return rows
