import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from src.models import VideoMetadata, RAGChunk, DatasetStats

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "tw3k_rag.db"


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Get SQLite connection with Row factory enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize SQLite database tables and Full-Text Search (FTS5) indices."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Videos table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            channel TEXT,
            upload_date TEXT,
            duration INTEGER,
            description TEXT,
            language TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # RAG Chunks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            formatted_time TEXT NOT NULL,
            timestamp_link TEXT NOT NULL,
            char_count INTEGER NOT NULL,
            word_count INTEGER NOT NULL,
            estimated_tokens INTEGER NOT NULL,
            FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
        );
    """)

    # FTS5 virtual table for lightning-fast keyword search
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED,
            video_id UNINDEXED,
            video_title,
            channel,
            text
        );
    """)

    conn.commit()
    conn.close()


def is_video_processed(video_id: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Check if a video ID is already stored in the SQLite database."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM videos WHERE video_id = ? LIMIT 1;", (video_id,))
    row = cursor.fetchone()
    conn.close()
    return row is not None


def save_video_and_chunks(
    metadata: VideoMetadata,
    chunks: List[RAGChunk],
    db_path: str = DEFAULT_DB_PATH
) -> None:

    """Save or replace video metadata and its RAG chunks in SQLite."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    try:
        # Upsert Video Metadata
        cursor.execute("""
            INSERT INTO videos (video_id, title, url, channel, upload_date, duration, description, language)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                title = excluded.title,
                url = excluded.url,
                channel = excluded.channel,
                upload_date = excluded.upload_date,
                duration = excluded.duration,
                description = excluded.description,
                language = excluded.language,
                processed_at = CURRENT_TIMESTAMP;
        """, (
            metadata.video_id, metadata.title, metadata.url, metadata.channel,
            metadata.upload_date, metadata.duration, metadata.description, metadata.language
        ))

        # Delete existing chunks for this video if updating
        cursor.execute("DELETE FROM chunks WHERE video_id = ?", (metadata.video_id,))
        cursor.execute("DELETE FROM chunks_fts WHERE video_id = ?", (metadata.video_id,))

        # Insert Chunks
        chunk_rows = []
        fts_rows = []
        for c in chunks:
            chunk_rows.append((
                c.chunk_id, c.video_id, c.chunk_index, c.text,
                c.start_time, c.end_time, c.formatted_time, c.timestamp_link,
                c.char_count, c.word_count, c.estimated_tokens
            ))
            fts_rows.append((c.chunk_id, c.video_id, metadata.title, metadata.channel, c.text))

        cursor.executemany("""
            INSERT INTO chunks (
                chunk_id, video_id, chunk_index, text, start_time, end_time,
                formatted_time, timestamp_link, char_count, word_count, estimated_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, chunk_rows)

        cursor.executemany("""
            INSERT INTO chunks_fts (chunk_id, video_id, video_title, channel, text)
            VALUES (?, ?, ?, ?, ?);
        """, fts_rows)

        conn.commit()
        logger.info(f"Saved video '{metadata.title}' with {len(chunks)} chunks to SQLite.")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def search_chunks(query: str, db_path: str = DEFAULT_DB_PATH, limit: int = 10) -> List[Dict[str, Any]]:
    """Search chunks using SQLite FTS5 full text search or LIKE fallback."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    results = []
    try:
        # FTS query search
        cursor.execute("""
            SELECT c.*, v.title as video_title, v.channel as channel
            FROM chunks_fts fts
            JOIN chunks c ON fts.chunk_id = c.chunk_id
            JOIN videos v ON c.video_id = v.video_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?;
        """, (query, limit))
        rows = cursor.fetchall()
        for r in rows:
            results.append(dict(r))
    except sqlite3.OperationalError:
        # Fallback to LIKE search if FTS query syntax error occurs
        cursor.execute("""
            SELECT c.*, v.title as video_title, v.channel as channel
            FROM chunks c
            JOIN videos v ON c.video_id = v.video_id
            WHERE c.text LIKE ? OR v.title LIKE ?
            LIMIT ?;
        """, (f"%{query}%", f"%{query}%", limit))
        rows = cursor.fetchall()
        for r in rows:
            results.append(dict(r))
    finally:
        conn.close()

    return results


def get_all_chunks(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Fetch all chunks joined with video metadata."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.*, v.title as video_title, v.channel, v.url as video_url
        FROM chunks c
        JOIN videos v ON c.video_id = v.video_id
        ORDER BY v.title, c.chunk_index;
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_videos(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Fetch all stored video records."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT v.*, COUNT(c.chunk_id) as chunk_count
        FROM videos v
        LEFT JOIN chunks c ON v.video_id = c.video_id
        GROUP BY v.video_id
        ORDER BY v.processed_at DESC;
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_dataset_stats(db_path: str = DEFAULT_DB_PATH) -> DatasetStats:
    """Compute overall stats for the SQLite database."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM videos;")
    total_videos = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*), SUM(word_count), AVG(char_count) FROM chunks;")
    row = cursor.fetchone()
    total_chunks = row[0] or 0
    total_words = row[1] or 0
    avg_chunk_chars = row[2] or 0.0

    conn.close()
    return DatasetStats(
        total_videos=total_videos,
        total_chunks=total_chunks,
        total_words=total_words,
        avg_chunk_chars=round(avg_chunk_chars, 1)
    )


def export_to_json(db_path: str = DEFAULT_DB_PATH, output_file: str = "dataset_export.json") -> str:
    """Export database into structured JSON format."""
    chunks = get_all_chunks(db_path)
    videos = get_all_videos(db_path)
    export_data = {
        "metadata": {
            "version": "1.0",
            "stats": get_dataset_stats(db_path).model_dump()
        },
        "videos": videos,
        "chunks": chunks
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    return str(Path(output_file).resolve())


def export_to_jsonl(db_path: str = DEFAULT_DB_PATH, output_file: str = "dataset_export.jsonl") -> str:
    """Export chunks into JSONL (JSON Lines) format - standard for RAG vector embeddings."""
    chunks = get_all_chunks(db_path)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return str(Path(output_file).resolve())


def export_to_csv(db_path: str = DEFAULT_DB_PATH, output_file: str = "dataset_export.csv") -> str:
    """Export chunks into CSV format using pandas."""
    chunks = get_all_chunks(db_path)
    df = pd.DataFrame(chunks)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    return str(Path(output_file).resolve())
