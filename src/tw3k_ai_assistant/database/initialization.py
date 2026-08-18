import os
from datetime import datetime
from typing import Any

import psycopg

DB_TIMEZONE = datetime.now().astimezone().tzinfo

DEFAULT_POSTGRES_HOST = "localhost"
DEFAULT_POSTGRES_PORT = "5432"
DEFAULT_POSTGRES_DB = "tw3k_monitoring"
DEFAULT_POSTGRES_USER = "tw3k"
DEFAULT_POSTGRES_PASSWORD = "password"


def get_db_connection() -> Any:
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", DEFAULT_POSTGRES_HOST),
        port=os.getenv("POSTGRES_PORT", DEFAULT_POSTGRES_PORT),
        dbname=os.getenv("POSTGRES_DB", DEFAULT_POSTGRES_DB),
        user=os.getenv("POSTGRES_USER", DEFAULT_POSTGRES_USER),
        password=os.getenv("POSTGRES_PASSWORD", DEFAULT_POSTGRES_PASSWORD),
    )


def init_feedback():
    """Create or migrate the feedback table to user scores only."""

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                    score INTEGER NOT NULL CHECK (score IN (-1, 1)),
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)

            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'feedback'
                """
            )
            columns = {row[0] for row in cur.fetchall()}
            if "source" in columns:
                cur.execute(
                    """
                    DELETE FROM feedback
                    WHERE source IS DISTINCT FROM 'user' OR score IS NULL
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE feedback
                    DROP COLUMN IF EXISTS source,
                    DROP COLUMN IF EXISTS relevance,
                    DROP COLUMN IF EXISTS explanation
                    """
                )

            cur.execute("ALTER TABLE feedback ALTER COLUMN conversation_id SET NOT NULL")
            cur.execute("ALTER TABLE feedback ALTER COLUMN score SET NOT NULL")
            cur.execute(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_constraint
                        WHERE conname = 'feedback_score_check'
                          AND conrelid = 'feedback'::regclass
                    ) THEN
                        ALTER TABLE feedback
                        ADD CONSTRAINT feedback_score_check CHECK (score IN (-1, 1));
                    END IF;
                END $$
                """
            )
        conn.commit()
    finally:
        conn.close()


def init_db(drop=False):
    """Create the conversations table without destroying existing rows.

    ``drop`` is retained for explicit local maintenance callers. The normal
    initializer and Compose service never pass it, so the default path is
    safe to run repeatedly.
    """

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if drop:
                cur.execute("DROP TABLE IF EXISTS feedback")
                cur.execute("DROP TABLE IF EXISTS conversations")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model TEXT NOT NULL,
                    instructions TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    response_time FLOAT NOT NULL,
                    cost FLOAT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    try:
        init_db()
        init_feedback()
    except Exception as exc:
        raise SystemExit(f"Database initialization failed: {exc}") from exc

    print("Database initialized")


if __name__ == "__main__":
    main()
