import unittest
import os
import json
import sqlite3
from pathlib import Path

from src.models import VideoMetadata, TranscriptSegment, format_seconds, build_youtube_timestamp_link
from src.youtube_fetcher import extract_video_id
from src.chunker import chunk_transcript
from src.storage import (
    init_db, save_video_and_chunks, search_chunks, get_dataset_stats,
    export_to_json, export_to_jsonl, export_to_csv, get_connection,
    is_video_processed
)


TEST_DB_PATH = "test_tw3k_rag.db"


class TestTW3KRAGPipeline(unittest.TestCase):

    def setUp(self):
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    def tearDown(self):
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        for ext in ["test_export.json", "test_export.jsonl", "test_export.csv"]:
            if os.path.exists(ext):
                os.remove(ext)

    def test_extract_video_id(self):
        """Test extraction of video ID from various YouTube URL formats."""
        url1 = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        url2 = "https://youtu.be/dQw4w9WgXcQ"
        url3 = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        plain = "dQw4w9WgXcQ"

        self.assertEqual(extract_video_id(url1), "dQw4w9WgXcQ")
        self.assertEqual(extract_video_id(url2), "dQw4w9WgXcQ")
        self.assertEqual(extract_video_id(url3), "dQw4w9WgXcQ")
        self.assertEqual(extract_video_id(plain), "dQw4w9WgXcQ")

    def test_timestamp_formatting(self):
        """Test time formatting and deep link construction."""
        self.assertEqual(format_seconds(65), "01:05")
        self.assertEqual(format_seconds(3665), "01:01:05")
        self.assertEqual(
            build_youtube_timestamp_link("ABC12345678", 125.4),
            "https://www.youtube.com/watch?v=ABC12345678&t=125s"
        )

    def test_transcript_chunking(self):
        """Test RAG transcript sliding window chunker."""
        segments = [
            TranscriptSegment(text="Welcome to Total War Three Kingdoms guide.", start=0.0, duration=3.0),
            TranscriptSegment(text="Liu Bei starts in Dong commandery with limited military units.", start=3.0, duration=4.0),
            TranscriptSegment(text="To fix his economy, build agriculture and tax collector buildings.", start=7.0, duration=5.0),
            TranscriptSegment(text="Form alliances with Gongsun Zan early in the campaign.", start=12.0, duration=4.0)
        ]
        metadata = VideoMetadata(
            video_id="test_vid_1",
            title="Liu Bei Beginner Guide",
            url="https://youtube.com/watch?v=test_vid_1",
            channel="Serious Trivia"
        )

        chunks = chunk_transcript(segments, metadata, target_chunk_size=100, chunk_overlap=20)
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks[0].video_id, "test_vid_1")
        self.assertIn("Total War Three Kingdoms", chunks[0].text)
        self.assertTrue(chunks[0].timestamp_link.startswith("https://www.youtube.com/watch?v=test_vid_1&t="))

    def test_sqlite_storage_and_fts_search(self):
        """Test SQLite saving, FTS query searching, and dataset exports."""
        init_db(TEST_DB_PATH)
        metadata = VideoMetadata(
            video_id="vid_fts_test",
            title="Cao Cao Strategy Guide",
            url="https://youtube.com/watch?v=vid_fts_test",
            channel="Total War Official"
        )
        segments = [
            TranscriptSegment(text="Cao Cao uses proxy wars to manipulate diplomacy.", start=10.0, duration=5.0),
            TranscriptSegment(text="Credibility is Cao Cao unique faction mechanic.", start=15.0, duration=5.0)
        ]
        chunks = chunk_transcript(segments, metadata, target_chunk_size=300, chunk_overlap=50)

        # Verify not processed before saving
        self.assertFalse(is_video_processed("vid_fts_test", db_path=TEST_DB_PATH))

        # Save
        save_video_and_chunks(metadata, chunks, db_path=TEST_DB_PATH)

        # Verify is_video_processed returns True after saving
        self.assertTrue(is_video_processed("vid_fts_test", db_path=TEST_DB_PATH))

        # Query stats
        stats = get_dataset_stats(TEST_DB_PATH)
        self.assertEqual(stats.total_videos, 1)
        self.assertEqual(stats.total_chunks, len(chunks))

        # Full-Text Search
        results = search_chunks("proxy wars", db_path=TEST_DB_PATH)
        self.assertEqual(len(results), 1)
        self.assertIn("proxy wars", results[0]['text'])

        # Exports
        json_path = export_to_json(TEST_DB_PATH, "test_export.json")
        jsonl_path = export_to_jsonl(TEST_DB_PATH, "test_export.jsonl")
        csv_path = export_to_csv(TEST_DB_PATH, "test_export.csv")

        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(jsonl_path))
        self.assertTrue(os.path.exists(csv_path))



if __name__ == "__main__":
    unittest.main()
