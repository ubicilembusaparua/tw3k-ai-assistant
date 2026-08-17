import os
import psycopg
from datetime import datetime

DB_TIMEZONE = datetime.now().astimezone().tzinfo
print(f"Using timezone: {DB_TIMEZONE}")

def get_db_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        dbname=os.getenv("POSTGRES_DB", "tw3k_monitoring"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
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
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if drop:
                cur.execute("DROP TABLE IF EXISTS conversations")

            cur.execute("""
                CREATE TABLE conversations (
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

if __name__ == "__main__":
    init_db()
    init_feedback()
    print("Database initialized")
