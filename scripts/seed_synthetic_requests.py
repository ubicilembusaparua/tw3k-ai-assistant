"""Replace the synthetic demo data with unique synthetic requests and feedback."""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
from tw3k_ai_assistant.config import load_environment
from tw3k_ai_assistant.database.initialization import get_db_connection


REQUEST_MODEL = "gpt-5.4-mini"
SEEDED_QUESTION_PREFIX = "[Synthetic request "
TOPICS = (
    "public order",
    "corruption reduction",
    "food production",
    "trade agreements",
    "spy networks",
    "army composition",
    "cavalry flanking",
    "siege defense",
    "character satisfaction",
    "diplomatic coalitions",
    "industrial income",
    "campaign movement",
    "night battles",
    "general recruitment",
    "commandery administration",
    "faction prestige",
    "crossbow formations",
    "settlement construction",
    "Han diplomacy",
    "replenishing casualties",
)


def build_synthetic_requests(
    count: int = 100,
    *,
    seed: int = 42,
    now: datetime | None = None,
) -> list[dict[str, object]]:
    """Build unique conversation rows with deterministic demo metrics."""

    if count <= 0:
        raise ValueError("Request count must be positive")

    generator = random.Random(seed)
    timestamp = now or datetime.now(timezone.utc)
    rows = []

    for index in range(count):
        request_number = index + 1
        topic = TOPICS[index % len(TOPICS)]
        variant = index // len(TOPICS) + 1
        question = (
            f"[Synthetic request {request_number:03d}] "
            f"What is a practical Total War: Three Kingdoms strategy for {topic} "
            f"in campaign scenario {variant}?"
        )
        answer = (
            f"Synthetic answer {request_number:03d}: prioritize the documented "
            f"{topic} mechanics, compare the available options, and choose the "
            "lowest-risk campaign action."
        )
        prompt = (
            f"QUESTION: {question}\n\n"
            f"CONTEXT:\nSynthetic dashboard context for {topic}, scenario {variant}."
        )
        prompt_tokens = generator.randint(450, 1800)
        completion_tokens = generator.randint(80, 320)

        rows.append(
            {
                "question": question,
                "answer": answer,
                "model": REQUEST_MODEL,
                "instructions": "Synthetic request generated for dashboard testing.",
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "response_time": round(generator.uniform(0.35, 4.5), 3),
                "cost": (prompt_tokens * 0.15 + completion_tokens * 0.60) / 1_000_000,
                "timestamp": timestamp - timedelta(hours=count - request_number) + timedelta(seconds=index),
            }
        )

    return rows


def build_synthetic_feedback(
    conversation_ids: list[int],
    request_rows: list[dict[str, object]],
    *,
    seed: int = 42,
) -> list[tuple[int, int, datetime]]:
    """Create one unique user vote per request."""

    if len(conversation_ids) != len(request_rows):
        raise ValueError("Conversation IDs and request rows must have equal length")

    generator = random.Random(seed)
    feedback_rows = []
    for conversation_id, request in zip(conversation_ids, request_rows):
        request_timestamp = request["timestamp"]
        assert isinstance(request_timestamp, datetime)

        score = generator.choices((1, -1), weights=(0.75, 0.25), k=1)[0]
        feedback_rows.append(
            (
                conversation_id,
                score,
                request_timestamp + timedelta(microseconds=1),
            )
        )

    return feedback_rows


def replace_synthetic_data(count: int = 100, *, seed: int = 42) -> tuple[int, int]:
    """Delete prior synthetic feedback, preserve real requests, and seed new data."""

    request_rows = build_synthetic_requests(count=count, seed=seed)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # The previous feedback-only seed produced no synthetic marker, so
            # clear the current feedback table as explicitly requested. Any
            # Seeded conversations use a question prefix as their private
            # replacement marker; the model label remains the real model name.
            seeded_question_pattern = f"{SEEDED_QUESTION_PREFIX}%"
            cur.execute(
                """
                DELETE FROM feedback
                WHERE conversation_id IN (
                    SELECT id FROM conversations WHERE question LIKE %s
                )
                """,
                (seeded_question_pattern,),
            )
            cur.execute(
                "DELETE FROM conversations WHERE question LIKE %s",
                (seeded_question_pattern,),
            )

            conversation_ids = []
            for request in request_rows:
                cur.execute(
                    """
                    INSERT INTO conversations (
                        question, answer, model, instructions, prompt,
                        prompt_tokens, completion_tokens, total_tokens,
                        response_time, cost, timestamp
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        request["question"],
                        request["answer"],
                        request["model"],
                        request["instructions"],
                        request["prompt"],
                        request["prompt_tokens"],
                        request["completion_tokens"],
                        request["total_tokens"],
                        request["response_time"],
                        request["cost"],
                        request["timestamp"],
                    ),
                )
                conversation_ids.append(cur.fetchone()[0])

            feedback_rows = build_synthetic_feedback(
                conversation_ids,
                request_rows,
                seed=seed,
            )
            cur.executemany(
                """
                INSERT INTO feedback (
                    conversation_id, score, timestamp
                ) VALUES (%s, %s, %s)
                """,
                feedback_rows,
            )

            cur.execute(
                """
                SELECT COUNT(*), COUNT(DISTINCT question), COUNT(DISTINCT prompt),
                       COUNT(DISTINCT timestamp)
                FROM conversations
                WHERE question LIKE %s
                """,
                (seeded_question_pattern,),
            )
            request_count, unique_questions, unique_prompts, unique_timestamps = cur.fetchone()
            if (request_count, unique_questions, unique_prompts, unique_timestamps) != (
                count,
                count,
                count,
                count,
            ):
                raise RuntimeError("Synthetic request uniqueness validation failed")

            cur.execute(
                """
                SELECT COUNT(*), COUNT(DISTINCT conversation_id), COUNT(DISTINCT timestamp)
                FROM feedback
                WHERE conversation_id = ANY(%s)
                """,
                (conversation_ids,),
            )
            feedback_count, unique_conversations, unique_feedback_timestamps = cur.fetchone()
            if (feedback_count, unique_conversations, unique_feedback_timestamps) != (
                count,
                count,
                count,
            ):
                raise RuntimeError("Synthetic feedback uniqueness validation failed")

        conn.commit()
        return count, count
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
        help="Required safety flag: replace prior synthetic/demo data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.replace:
        raise SystemExit("Refusing to modify data without --replace")

    load_environment(ROOT_DIR / ".env")
    requests, feedback = replace_synthetic_data(count=args.count, seed=args.seed)
    print(f"Seeded {requests} synthetic requests and {feedback} feedback rows.")


if __name__ == "__main__":
    main()
