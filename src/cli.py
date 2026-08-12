import sys
import logging

# Ensure UTF-8 stdout on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn
import time
import random


from src.youtube_fetcher import fetch_video_metadata, fetch_transcript, extract_playlist_video_urls, extract_video_id
from src.chunker import chunk_transcript
from src.storage import (
    save_video_and_chunks, search_chunks, get_all_videos, get_all_chunks,
    get_dataset_stats, export_to_json, export_to_jsonl, export_to_csv,
    DEFAULT_DB_PATH, is_video_processed
)

console = Console()
logging.basicConfig(level=logging.WARNING)


def print_banner():
    console.print(Panel.fit(
        "[bold gold1]Total War: Three Kingdoms[/bold gold1] - [bold cyan]YouTube RAG Dataset Builder[/bold cyan]\n"
        "[dim]Convert YouTube videos into timestamped RAG-ready datasets (SQLite / CSV / JSON)[/dim]",
        border_style="cyan"
    ))


def process_single_video(url_or_id: str, chunk_size: int = 500, overlap: int = 100, db_path: str = DEFAULT_DB_PATH, force_reprocess: bool = False):
    """Process a single YouTube video URL into SQLite RAG chunks with deduplication."""
    try:
        vid = extract_video_id(url_or_id)
        if not force_reprocess and is_video_processed(vid, db_path=db_path):
            console.print(f"[yellow]⏩ Video ID '{vid}' is already processed in database. Skipping.[/yellow]")
            return
    except Exception:
        pass

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    ) as progress:
        task1 = progress.add_task("[cyan]Fetching video metadata from YouTube...", total=None)
        try:
            metadata = fetch_video_metadata(url_or_id)
        except Exception as e:
            console.print(f"[bold red]Error fetching metadata:[/bold red] {e}")
            return

        progress.update(task1, description="[cyan]Fetching timestamped transcripts...")
        try:
            segments = fetch_transcript(metadata.video_id)
        except Exception as e:
            console.print(f"[bold red]Error fetching transcript for '{metadata.title}':[/bold red] {e}")
            return

        progress.update(task1, description="[cyan]Chunking transcript for RAG ingestion...")
        chunks = chunk_transcript(segments, metadata, target_chunk_size=chunk_size, chunk_overlap=overlap)

        progress.update(task1, description="[cyan]Saving to SQLite database...")
        save_video_and_chunks(metadata, chunks, db_path=db_path)

    console.print(f"\n[bold green]✓ Successfully processed video:[/bold green] [bold white]{metadata.title}[/bold white]")
    console.print(f"  • Video ID: [yellow]{metadata.video_id}[/yellow]")
    console.print(f"  • Channel: [cyan]{metadata.channel}[/cyan]")
    console.print(f"  • Raw Segments: [magenta]{len(segments)}[/magenta]")
    console.print(f"  • RAG Chunks Generated: [green]{len(chunks)}[/green]\n")

    if chunks:
        console.print("[dim]Sample Chunk Preview (Chunk #0):[/dim]")
        sample = chunks[0]
        console.print(Panel(
            f"[bold yellow]Timestamp:[/bold yellow] {sample.formatted_time} ([link={sample.timestamp_link}]Open in YouTube[/link])\n"
            f"[bold yellow]Text:[/bold yellow] {sample.text}",
            title=f"Chunk ID: {sample.chunk_id}",
            border_style="dim white"
        ))


def process_playlist(playlist_url: str, chunk_size: int = 500, overlap: int = 100, db_path: str = DEFAULT_DB_PATH):
    """Process an entire YouTube playlist URL with rate-limiting delays and deduplication."""
    console.print(f"[cyan]Extracting video URLs from playlist...[/cyan]")
    try:
        videos = extract_playlist_video_urls(playlist_url)
    except Exception as e:
        console.print(f"[bold red]Error parsing playlist:[/bold red] {e}")
        return

    console.print(f"Found [bold yellow]{len(videos)}[/bold yellow] videos in playlist.")
    for idx, (vid, url) in enumerate(videos, start=1):
        if is_video_processed(vid, db_path=db_path):
            console.print(f"[yellow]⏩ [{idx}/{len(videos)}] Video '{vid}' already in database. Skipping.[/yellow]")
            continue

        console.print(f"\n[bold cyan]Processing [{idx}/{len(videos)}]:[/bold cyan] {url}")
        process_single_video(url, chunk_size=chunk_size, overlap=overlap, db_path=db_path)

        # Delay between video processing to avoid rate-limiting
        if idx < len(videos):
            delay = random.uniform(1.0, 3.0)
            time.sleep(delay)



def handle_search(db_path: str = DEFAULT_DB_PATH):
    """Interactive FTS keyword search."""
    query = Prompt.ask("\n[bold yellow]Enter search query for TW3K dataset[/bold yellow]")
    if not query.strip():
        return

    results = search_chunks(query, db_path=db_path, limit=10)
    if not results:
        console.print(f"[bold red]No matching chunks found for query:[/bold red] '{query}'")
        return

    console.print(f"\n[bold green]Found {len(results)} matching chunks:[/bold green]")
    for r in results:
        console.print(Panel(
            f"[bold yellow]Video:[/bold yellow] {r['video_title']} ({r['channel']})\n"
            f"[bold yellow]Time:[/bold yellow] {r['formatted_time']} | [bold blue]Link:[/bold blue] {r['timestamp_link']}\n"
            f"[bold white]Content:[/bold white] {r['text']}",
            title=f"Chunk: {r['chunk_id']}",
            border_style="green"
        ))


def handle_inspect_db(db_path: str = DEFAULT_DB_PATH):
    """Inspect stored database videos and stats."""
    stats = get_dataset_stats(db_path)
    console.print("\n[bold cyan]=== Dataset Summary Statistics ===[/bold cyan]")
    console.print(f"• Total Videos Processed: [bold yellow]{stats.total_videos}[/bold yellow]")
    console.print(f"• Total RAG Chunks: [bold yellow]{stats.total_chunks}[/bold yellow]")
    console.print(f"• Total Words: [bold yellow]{stats.total_words:,}[/bold yellow]")
    console.print(f"• Avg Chunk Character Length: [bold yellow]{stats.avg_chunk_chars:.1f}[/bold yellow] chars\n")

    videos = get_all_videos(db_path)
    if videos:
        table = Table(title="Processed Videos in SQLite Database", border_style="cyan")
        table.add_column("Video ID", style="yellow")
        table.add_column("Title", style="white")
        table.add_column("Channel", style="cyan")
        table.add_column("Chunks", style="magenta", justify="right")
        table.add_column("Processed At", style="dim")

        for v in videos:
            table.add_row(
                v['video_id'],
                v['title'][:40] + ("..." if len(v['title']) > 40 else ""),
                v['channel'],
                str(v['chunk_count']),
                str(v['processed_at'])
            )
        console.print(table)


def handle_export(db_path: str = DEFAULT_DB_PATH):
    """Handle database export options."""
    console.print("\n[bold cyan]Export Options:[/bold cyan]")
    console.print("1. Export to JSON (.json)")
    console.print("2. Export to JSONL (.jsonl - recommended for RAG vector DBs)")
    console.print("3. Export to CSV (.csv)")
    console.print("4. Export ALL formats")

    choice = Prompt.ask("Select format option", choices=["1", "2", "3", "4"], default="4")

    if choice in ["1", "4"]:
        path = export_to_json(db_path, "tw3k_dataset.json")
        console.print(f"[bold green]✓ Exported JSON to:[/bold green] {path}")
    if choice in ["2", "4"]:
        path = export_to_jsonl(db_path, "tw3k_dataset.jsonl")
        console.print(f"[bold green]✓ Exported JSONL to:[/bold green] {path}")
    if choice in ["3", "4"]:
        path = export_to_csv(db_path, "tw3k_dataset.csv")
        console.print(f"[bold green]✓ Exported CSV to:[/bold green] {path}")


def run_cli_menu():
    """Main CLI loop."""
    print_banner()
    db_path = DEFAULT_DB_PATH

    while True:
        console.print("\n[bold gold1]Main Menu:[/bold gold1]")
        console.print("1. [bold white]Process a YouTube Video URL[/bold white]")
        console.print("2. [bold white]Process a YouTube Playlist URL[/bold white]")
        console.print("3. [bold white]Search Dataset (SQLite Full-Text Search)[/bold white]")
        console.print("4. [bold white]Inspect Stored Database & Stats[/bold white]")
        console.print("5. [bold white]Export Dataset (JSON / JSONL / CSV)[/bold white]")
        console.print("6. [bold red]Exit[/bold red]")

        choice = Prompt.ask("\nChoose an option", choices=["1", "2", "3", "4", "5", "6"], default="1")

        if choice == "1":
            url = Prompt.ask("Enter YouTube Video URL or Video ID")
            chunk_size = IntPrompt.ask("Target Chunk Size (characters)", default=500)
            overlap = IntPrompt.ask("Chunk Overlap (characters)", default=100)
            process_single_video(url, chunk_size=chunk_size, overlap=overlap, db_path=db_path)
        elif choice == "2":
            url = Prompt.ask("Enter YouTube Playlist URL")
            chunk_size = IntPrompt.ask("Target Chunk Size (characters)", default=500)
            overlap = IntPrompt.ask("Chunk Overlap (characters)", default=100)
            process_playlist(url, chunk_size=chunk_size, overlap=overlap, db_path=db_path)
        elif choice == "3":
            handle_search(db_path=db_path)
        elif choice == "4":
            handle_inspect_db(db_path=db_path)
        elif choice == "5":
            handle_export(db_path=db_path)
        elif choice == "6":
            console.print("[yellow]Exiting TW3K YouTube RAG Dataset Builder. Goodbye![/yellow]")
            sys.exit(0)


if __name__ == "__main__":
    run_cli_menu()
