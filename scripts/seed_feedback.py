"""Replace the feedback table with unique synthetic feedback rows."""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db_init import get_db_connection


JUDGE_RELEVANCE = ("RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT")


def build_feedback_rows(
    conversation_ids: list[int],
    count: int,
    *,
    seed: int = 42,
    now: datetime | None = None,
) -> list[tuple[int, str, str | None, str, int | None, datetime]]:
    """Build distinct synthetic rows distributed across existing conversations."""

    if not conversation_ids:
        raise ValueError("At least one conversation is required")
    if count <= 0:
        raise ValueError("Feedback count must be positive")

    generator = random.Random(seed)
    timestamp = now or datetime.now(timezone.utc)
    rows = []

    for index in range(count):
        feedback_number = index + 1
        conversation_id = conversation_ids[index % len(conversation_ids)]
        row_timestamp = timestamp - timedelta(seconds=count - feedback_number)

        if index % 2 == 0:
            relevance = generator.choices(
                JUDGE_RELEVANCE,
                weights=(0.60, 0.25, 0.15),
                k=1,
            )[0]
            rows.append(
                (
                    conversation_id,
                    "judge",
                    relevance,
                    f"Synthetic judge evaluation #{feedback_number:03d}: {relevance.lower()} response.",
                    None,
                    row_timestamp,
                )
            )
        else:
            score = generator.choices((1, -1), weights=(0.75, 0.25), k=1)[0]
            rows.append(
                (
                    conversation_id,
                    "user",
                    None,
                    f"Synthetic user vote #{feedback_number:03d}: score {score:+d}.",
                    score,
                    row_timestamp,
                )
            )

    return rows


def replace_feedback(count: int = 100, *, seed: int = 42) -> int:
    """Delete existing feedback and insert a fresh synthetic dataset atomically."""

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM conversations ORDER BY id")
            conversation_ids = [row[0] for row in cur.fetchall()]
            rows = build_feedback_rows(conversation_ids, count, seed=seed)

            cur.execute("DELETE FROM feedback")
            cur.executemany(
                """
                INSERT INTO feedback (
                    conversation_id, source, relevance,
                    explanation, score, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                rows,
            )
            cur.execute(
                """
                SELECT COUNT(*), COUNT(DISTINCT timestamp), COUNT(DISTINCT explanation)
                FROM feedback
                """
            )
            inserted, unique_timestamps, unique_explanations = cur.fetchone()

            if inserted != count or unique_timestamps != count or unique_explanations != count:
                raise RuntimeError("Synthetic feedback uniqueness validation failed")

        conn.commit()
        return inserted
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Required safety flag: delete existing feedback before seeding.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.replace:
        raise SystemExit("Refusing to modify feedback without --replace")

    load_dotenv(ROOT_DIR / ".env")
    inserted = replace_feedback(count=args.count, seed=args.seed)
    print(f"Replaced feedback with {inserted} unique synthetic rows.")


if __name__ == "__main__":
    main()
