from datetime import datetime
from db_init import get_db_connection, DB_TIMEZONE


def save_feedback(conversation_id, score):
    if score not in (-1, 1):
        raise ValueError("User feedback score must be 1 or -1")

    timestamp = datetime.now(DB_TIMEZONE)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (
                    conversation_id, score, timestamp
                ) VALUES (
                    %s, %s, %s
                )
                """,
                (conversation_id, score, timestamp),
            )
        conn.commit()
    finally:
        conn.close()
