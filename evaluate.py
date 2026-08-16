"""Fast Side-by-side Retrieval Evaluation (Top 10 Results) pulling existing vectors from Qdrant DB."""

from src.dataset import load_dataset
from src.bm25_retriever import BM25Retriever
from src.qdrant_retriever import QdrantRetriever
from src.hybrid_retriever import HybridRetriever


def main():
    print("==================================================")
    print(" Full Dataset Retrieval Evaluation (Top 10 Results)")
    print("==================================================\n")

    # 1. Load dataset for BM25
    print("Loading transcript chunks from tw3k_dataset.jsonl...")
    chunks = load_dataset("tw3k_dataset.jsonl")
    print(f"Loaded {len(chunks)} transcript chunks.\n")

    # 2. Initialize BM25 Lexical Retriever directly from dataset
    print("Building BM25 Lexical index...")
    bm25_retriever = BM25Retriever(chunks)

    # 3. Connect to Qdrant Vector DB (Pulls existing vectors without re-embedding if available)
    print("Connecting to Qdrant Vector Database...")
    qdrant_retriever = QdrantRetriever(collection_name="tw3k_transcripts")
    
    # Index only if Qdrant collection is currently empty
    if qdrant_retriever.get_point_count() == 0:
        print(f"Qdrant collection empty. Ingesting {len(chunks)} chunks...")
        qdrant_retriever.index_chunks(chunks, batch_size=64)
    else:
        print(f"Connected to Qdrant DB: {qdrant_retriever.get_point_count()} existing points found.")

    # 4. Initialize Hybrid RRF Retriever
    print("Initializing Hybrid RRF Fusion Retriever...")
    hybrid_retriever = HybridRetriever(bm25_retriever, qdrant_retriever, rrf_k=60)

    print("\n--------------------------------------------------")

    queries = [
        "How to manage public order and reduce corruption in commanderies?",
        "Cao Cao credibility mechanics and diplomatic proxy wars",
    ]

    for query in queries:
        print(f"\nQUERY: '{query}'")
        print("=" * 80)

        for name, retriever in [
            ("BM25 Lexical", bm25_retriever),
            ("Qdrant Vector DB", qdrant_retriever),
            ("Hybrid RRF", hybrid_retriever),
        ]:
            print(f"\n--- {name} (Top 10 Results) ---")
            results = retriever.search(query, top_k=10)

            for item in results:
                title = item.chunk.metadata.get("video_title", "Unknown")
                time_str = item.chunk.metadata.get("formatted_time", "")
                snippet = item.chunk.content[:100].replace("\n", " ")
                print(f"  Rank {item.rank:2d} | Score: {item.score:7.4f} | ID: {item.chunk.id} | Video: {title[:35]} ({time_str})")
                print(f"           Snippet: \"{snippet}...\"")


if __name__ == "__main__":
    main()
