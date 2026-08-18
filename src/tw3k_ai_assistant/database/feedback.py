from datetime import datetime
from tw3k_ai_assistant.database.initialization import DB_TIMEZONE, get_db_connection


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
