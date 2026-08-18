from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from tw3k_ai_assistant.database.initialization import get_db_connection
from tw3k_ai_assistant.rag.metrics import LLMCallRecord

@dataclass
class Stats:
    total: int
    avg_response_time: float
    total_cost: float
    avg_tokens: float


@dataclass(frozen=True)
class ConversationMetric:
    """A conversation row enriched with its latest user feedback."""

    id: int
    question: str
    answer: str
    model: str
    instructions: str
    prompt: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    timestamp: datetime
    user_score: Optional[int] = None

def get_stats():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*),
                    COALESCE(AVG(response_time), 0),
                    COALESCE(SUM(cost), 0),
                    COALESCE(AVG(total_tokens), 0)
                FROM conversations
            """)
            row = cur.fetchone()
    finally:
        conn.close()

    return Stats(
        total=int(row[0] or 0),
        avg_response_time=float(row[1] or 0),
        total_cost=float(row[2] or 0),
        avg_tokens=float(row[3] or 0),
    )

def row_to_record(row):
    return LLMCallRecord(
        model=row[3],
        prompt=row[5],
        instructions=row[4],
        answer=row[2],
        prompt_tokens=row[6],
        completion_tokens=row[7],
        total_tokens=row[8],
        response_time=row[9],
        cost=row[10],
        timestamp=row[11],
    )

def get_conversations(limit=10):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, question, answer, model,
                       instructions, prompt,
                       prompt_tokens, completion_tokens, total_tokens,
                       response_time, cost, timestamp
                FROM conversations
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [row_to_record(row) for row in rows]


def _row_to_conversation_metric(row) -> ConversationMetric:
    return ConversationMetric(
        id=row[0],
        question=row[1],
        answer=row[2],
        model=row[3],
        instructions=row[4],
        prompt=row[5],
        prompt_tokens=row[6],
        completion_tokens=row[7],
        total_tokens=row[8],
        response_time=row[9],
        cost=row[10],
        timestamp=row[11],
        user_score=row[12],
    )


def get_conversation_metrics(limit=1000) -> list[ConversationMetric]:
    """Return recent conversations with their latest user feedback."""

    if limit <= 0:
        return []

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.question, c.answer, c.model,
                       c.instructions, c.prompt,
                       c.prompt_tokens, c.completion_tokens, c.total_tokens,
                       c.response_time, c.cost, c.timestamp,
                       user_feedback.score
                FROM conversations AS c
                LEFT JOIN LATERAL (
                    SELECT f.score
                    FROM feedback AS f
                    WHERE f.conversation_id = c.id
                      AND f.score IS NOT NULL
                    ORDER BY f.timestamp DESC, f.id DESC
                    LIMIT 1
                ) AS user_feedback ON TRUE
                ORDER BY c.timestamp DESC
                LIMIT %s
                """,
                (int(limit),),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [_row_to_conversation_metric(row) for row in rows]

if __name__ == "__main__":
    records = get_conversations()
    for record in records:
        print(record)
