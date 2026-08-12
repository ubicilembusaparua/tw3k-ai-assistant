from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional

import time
import random

from src.youtube_fetcher import fetch_video_metadata, fetch_transcript, extract_playlist_video_urls
from src.chunker import chunk_transcript
from src.storage import (
    save_video_and_chunks, search_chunks, get_all_chunks, get_dataset_stats,
    export_to_json, export_to_jsonl, export_to_csv, DEFAULT_DB_PATH, init_db,
    is_video_processed
)

app = FastAPI(title="Total War: Three Kingdoms RAG Dataset Studio")

# Ensure DB initialized
init_db(DEFAULT_DB_PATH)


class ProcessRequest(BaseModel):
    url: str
    chunk_size: Optional[int] = 500
    chunk_overlap: Optional[int] = 100
    force_reprocess: Optional[bool] = False


@app.get("/", response_class=HTMLResponse)
def read_root():
    template_path = Path(__file__).parent.parent / "templates" / "index.html"
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Template not found")
    return template_path.read_text(encoding="utf-8")


@app.post("/api/process")
def process_video_endpoint(req: ProcessRequest):
    """API endpoint to ingest video or playlist URL into dataset with deduplication and delay."""
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    try:
        # Check if playlist URL
        if "playlist?list=" in url:
            videos = extract_playlist_video_urls(url)
            total_chunks = 0
            processed_count = 0
            skipped_count = 0
            failed_count = 0

            for idx, (vid, vurl) in enumerate(videos):
                # Deduplication check
                if not req.force_reprocess and is_video_processed(vid, DEFAULT_DB_PATH):
                    skipped_count += 1
                    continue

                try:
                    meta = fetch_video_metadata(vurl)
                    segs = fetch_transcript(meta.video_id)
                    chunks = chunk_transcript(segs, meta, target_chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
                    save_video_and_chunks(meta, chunks, db_path=DEFAULT_DB_PATH)
                    total_chunks += len(chunks)
                    processed_count += 1

                    # Delay between video processing to avoid rate-limiting
                    if idx < len(videos) - 1:
                        time.sleep(random.uniform(1.0, 3.0))

                except Exception as vid_err:
                    failed_count += 1
                    continue

            return {
                "status": "success",
                "message": f"Playlist processed: {processed_count} new, {skipped_count} skipped (already in DB), {failed_count} failed.",
                "processed_count": processed_count,
                "skipped_count": skipped_count,
                "failed_count": failed_count,
                "chunks_count": total_chunks
            }
        else:
            meta = fetch_video_metadata(url)
            if not req.force_reprocess and is_video_processed(meta.video_id, DEFAULT_DB_PATH):
                return {
                    "status": "skipped",
                    "message": f"Video '{meta.title}' is already in dataset.",
                    "video_id": meta.video_id
                }

            segs = fetch_transcript(meta.video_id)
            chunks = chunk_transcript(segs, meta, target_chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
            save_video_and_chunks(meta, chunks, db_path=DEFAULT_DB_PATH)
            return {
                "status": "success",
                "message": f"Processed video '{meta.title}'",
                "video_id": meta.video_id,
                "chunks_count": len(chunks)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/search")
def search_endpoint(q: str = Query(..., min_length=1), limit: int = 20):
    """API endpoint for SQLite Full-Text Search."""
    try:
        results = search_chunks(q, db_path=DEFAULT_DB_PATH, limit=limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chunks")
def get_chunks_endpoint(limit: int = 50):
    """Get stored RAG chunks."""
    chunks = get_all_chunks(DEFAULT_DB_PATH)
    return chunks[:limit]


@app.get("/api/stats")
def get_stats_endpoint():
    """Get dataset summary statistics."""
    return get_dataset_stats(DEFAULT_DB_PATH)


@app.get("/api/export/json")
def export_json_endpoint():
    file_path = export_to_json(DEFAULT_DB_PATH, "tw3k_dataset.json")
    return FileResponse(path=file_path, filename="tw3k_dataset.json", media_type="application/json")


@app.get("/api/export/jsonl")
def export_jsonl_endpoint():
    file_path = export_to_jsonl(DEFAULT_DB_PATH, "tw3k_dataset.jsonl")
    return FileResponse(path=file_path, filename="tw3k_dataset.jsonl", media_type="application/x-jsonlines")


@app.get("/api/export/csv")
def export_csv_endpoint():
    file_path = export_to_csv(DEFAULT_DB_PATH, "tw3k_dataset.csv")
    return FileResponse(path=file_path, filename="tw3k_dataset.csv", media_type="text/csv")
