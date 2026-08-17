from datetime import datetime, timezone

from scripts.seed_synthetic_requests import (
    build_synthetic_feedback,
    build_synthetic_requests,
)


def test_synthetic_requests_and_feedback_are_unique():
    requests = build_synthetic_requests(
        count=100,
        now=datetime(2026, 8, 17, tzinfo=timezone.utc),
    )
    feedback = build_synthetic_feedback(
        list(range(1, 101)),
        requests,
    )

    assert len(requests) == 100
    assert len({row["question"] for row in requests}) == 100
    assert len({row["prompt"] for row in requests}) == 100
    assert len({row["timestamp"] for row in requests}) == 100

    assert len(feedback) == 100
    assert len({row[0] for row in feedback}) == 100
    assert len({row[2] for row in feedback}) == 100
    assert all(row[1] in (-1, 1) for row in feedback)
