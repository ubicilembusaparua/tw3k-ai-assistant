import json
from pathlib import Path
from typing import List, Union
from tw3k_ai_assistant.retrieval.schema import DocumentChunk


DEFAULT_DATASET_PATH = Path("data/tw3k_dataset.jsonl")


def load_dataset(
    filepath: Union[str, Path] = DEFAULT_DATASET_PATH,
) -> List[DocumentChunk]:
    """Loads Total War: Three Kingdoms transcript passages from JSONL file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")

    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            chunk_id = data.get("chunk_id", str(len(chunks)))
            content = data.get("text", "")
            metadata = {
                "video_id": data.get("video_id"),
                "video_title": data.get("video_title"),
                "formatted_time": data.get("formatted_time"),
                "timestamp_link": data.get("timestamp_link"),
                "channel": data.get("channel"),
                "start_time": data.get("start_time"),
                "end_time": data.get("end_time"),
            }
            if content:
                chunks.append(DocumentChunk(id=chunk_id, content=content, metadata=metadata))

    return chunks
