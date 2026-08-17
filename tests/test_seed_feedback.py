from datetime import datetime, timezone

from scripts.seed_feedback import build_feedback_rows


def test_build_feedback_rows_are_unique_and_balanced_between_sources():
    rows = build_feedback_rows(
        [1, 2, 3],
        100,
        now=datetime(2026, 8, 17, tzinfo=timezone.utc),
    )

    assert len(rows) == 100
    assert len({row[3] for row in rows}) == 100
    assert len({row[5] for row in rows}) == 100
    assert {row[1] for row in rows} == {"judge", "user"}
    assert sum(row[1] == "judge" for row in rows) == 50
    assert sum(row[1] == "user" for row in rows) == 50
