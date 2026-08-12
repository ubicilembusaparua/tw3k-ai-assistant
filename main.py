import sys
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import uvicorn
from rich.console import Console


from src.cli import (
    run_cli_menu, process_single_video, process_playlist,
    handle_search, handle_inspect_db, handle_export
)
from src.storage import (
    DEFAULT_DB_PATH, search_chunks, export_to_json, export_to_jsonl, export_to_csv
)



console = Console()


def main():
    parser = argparse.ArgumentParser(description="Total War: Three Kingdoms - YouTube RAG Dataset Builder")
    parser.add_argument("--web", action="store_true", help="Launch interactive Web Dashboard (FastAPI)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web server host IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    
    parser.add_argument("--video", type=str, help="Ingest a single YouTube video URL or ID")
    parser.add_argument("--playlist", type=str, help="Ingest a YouTube playlist URL")
    parser.add_argument("--search", type=str, help="Search SQLite dataset with keyword query")
    parser.add_argument("--inspect", action="store_true", help="Inspect stored database statistics")
    parser.add_argument("--export", type=str, choices=["json", "jsonl", "csv", "all"], help="Export dataset format")
    parser.add_argument("--chunk-size", type=int, default=500, help="Target chunk size in characters")
    parser.add_argument("--overlap", type=int, default=100, help="Chunk overlap in characters")

    args = parser.parse_args()

    if args.web:
        console.print(f"[bold cyan]Launching Web Dashboard on http://{args.host}:{args.port}[/bold cyan]")
        uvicorn.run("src.web_app:app", host=args.host, port=args.port, reload=True)
    elif args.video:
        process_single_video(args.video, chunk_size=args.chunk_size, overlap=args.overlap, db_path=DEFAULT_DB_PATH)
    elif args.playlist:
        process_playlist(args.playlist, chunk_size=args.chunk_size, overlap=args.overlap, db_path=DEFAULT_DB_PATH)
    elif args.search:
        results = search_chunks(args.search, db_path=DEFAULT_DB_PATH)
        for r in results:
            console.print(f"[{r['chunk_id']}] {r['video_title']} @ {r['formatted_time']}\n{r['text']}\n")
    elif args.inspect:
        handle_inspect_db(db_path=DEFAULT_DB_PATH)
    elif args.export:
        if args.export == "json" or args.export == "all":
            p = export_to_json(DEFAULT_DB_PATH, "tw3k_dataset.json")
            console.print(f"[bold green]✓ Exported JSON to:[/bold green] {p}")
        if args.export == "jsonl" or args.export == "all":
            p = export_to_jsonl(DEFAULT_DB_PATH, "tw3k_dataset.jsonl")
            console.print(f"[bold green]✓ Exported JSONL to:[/bold green] {p}")
        if args.export == "csv" or args.export == "all":
            p = export_to_csv(DEFAULT_DB_PATH, "tw3k_dataset.csv")
            console.print(f"[bold green]✓ Exported CSV to:[/bold green] {p}")

    else:
        # Default to interactive CLI menu if no args given
        run_cli_menu()


if __name__ == "__main__":
    main()
