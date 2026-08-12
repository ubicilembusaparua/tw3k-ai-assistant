from typing import List, Optional
from pydantic import BaseModel, Field


def format_seconds(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_youtube_timestamp_link(video_id: str, start_seconds: float) -> str:
    """Generate clickable YouTube timestamp link."""
    start_sec_int = int(start_seconds)
    return f"https://www.youtube.com/watch?v={video_id}&t={start_sec_int}s"


class TranscriptSegment(BaseModel):
    """Raw subtitle segment extracted from YouTube video."""
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def formatted_start(self) -> str:
        return format_seconds(self.start)


class VideoMetadata(BaseModel):
    """YouTube video metadata details."""
    video_id: str
    title: str
    url: str
    channel: str = "Unknown Channel"
    upload_date: Optional[str] = None
    duration: Optional[int] = 0
    description: Optional[str] = ""
    language: str = "en"


class RAGChunk(BaseModel):
    """Chunk of text formatted and aggregated specifically for RAG ingestion."""
    chunk_id: str
    video_id: str
    video_title: str
    channel: str
    chunk_index: int
    text: str
    start_time: float
    end_time: float
    formatted_time: str
    timestamp_link: str
    char_count: int
    word_count: int
    estimated_tokens: int


class DatasetStats(BaseModel):
    """Summary statistics for the dataset."""
    total_videos: int = 0
    total_chunks: int = 0
    total_words: int = 0
    avg_chunk_chars: float = 0.0
