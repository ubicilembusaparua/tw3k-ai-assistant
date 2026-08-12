from typing import List
from src.models import TranscriptSegment, VideoMetadata, RAGChunk, format_seconds, build_youtube_timestamp_link


def chunk_transcript(
    segments: List[TranscriptSegment],
    metadata: VideoMetadata,
    target_chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[RAGChunk]:
    """
    RAG-optimized sliding window transcript chunker.
    
    Combines consecutive short transcript snippets into coherent text blocks
    while tracking timestamp metadata for deep-linking.
    
    Args:
        segments: List of raw TranscriptSegments from youtube-transcript-api.
        metadata: VideoMetadata object containing video title and channel.
        target_chunk_size: Approximate target character length per chunk (default 500).
        chunk_overlap: Number of overlapping characters retained between adjacent chunks (default 100).
        
    Returns:
        List of RAGChunk objects ready for database storage and vector embedding.
    """
    if not segments:
        return []

    chunks: List[RAGChunk] = []
    current_segment_group: List[TranscriptSegment] = []
    current_text_len = 0
    chunk_index = 0

    i = 0
    while i < len(segments):
        seg = segments[i]
        current_segment_group.append(seg)
        current_text_len += len(seg.text) + 1  # include space

        # Check if we reached target chunk size or last segment
        if current_text_len >= target_chunk_size or i == len(segments) - 1:
            # Build combined chunk text
            chunk_text = " ".join(s.text for s in current_segment_group).strip()
            
            start_time = current_segment_group[0].start
            end_time = current_segment_group[-1].end

            formatted_time = f"{format_seconds(start_time)} - {format_seconds(end_time)}"
            timestamp_link = build_youtube_timestamp_link(metadata.video_id, start_time)

            word_count = len(chunk_text.split())
            char_count = len(chunk_text)
            estimated_tokens = max(1, char_count // 4)

            rag_chunk = RAGChunk(
                chunk_id=f"{metadata.video_id}_c{chunk_index:04d}",
                video_id=metadata.video_id,
                video_title=metadata.title,
                channel=metadata.channel,
                chunk_index=chunk_index,
                text=chunk_text,
                start_time=round(start_time, 2),
                end_time=round(end_time, 2),
                formatted_time=formatted_time,
                timestamp_link=timestamp_link,
                char_count=char_count,
                word_count=word_count,
                estimated_tokens=estimated_tokens
            )
            chunks.append(rag_chunk)
            chunk_index += 1

            # Prepare for next window with overlap
            if i == len(segments) - 1:
                break

            # Calculate overlap by stepping backwards in segments
            overlap_len = 0
            new_group = []
            for prev_seg in reversed(current_segment_group):
                if overlap_len + len(prev_seg.text) <= chunk_overlap:
                    new_group.insert(0, prev_seg)
                    overlap_len += len(prev_seg.text) + 1
                else:
                    break

            current_segment_group = new_group
            current_text_len = sum(len(s.text) + 1 for s in current_segment_group)

        i += 1

    return chunks
