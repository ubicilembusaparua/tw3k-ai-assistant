"""Dedicated ingestion script to embed updated tw3k_dataset.jsonl and persist vectors into Qdrant Vector Database."""

from src.dataset import load_dataset
from src.qdrant_retriever import QdrantRetriever


def main():
    print("==================================================")
    print(" Qdrant Vector Database Ingestion Script")
    print("==================================================\n")

    # 1. Load latest dataset
    print("Loading transcript chunks from updated tw3k_dataset.jsonl...")
    chunks = load_dataset("tw3k_dataset.jsonl")
    print(f"Loaded {len(chunks)} document chunks.\n")

    # 2. Connect to Qdrant DB (Docker / Local server at localhost:6333)
    collection_name = "tw3k_transcripts"
    print(f"Connecting to Qdrant Vector DB (Collection: '{collection_name}')...")
    qdrant = QdrantRetriever(collection_name=collection_name, in_memory=False)

    # 3. Embed and upsert vectors (force=True purges old points and indexes fresh dataset)
    print(f"Embedding and indexing {len(chunks)} fresh chunks using local ONNX Embedder into Qdrant...")
    qdrant.index_chunks(chunks, batch_size=64, force=True)

    # 4. Verify point count
    try:
        count = qdrant.get_point_count()
        print(f"\nSUCCESS: Ingestion complete! Total points stored in Qdrant DB: {count}")
    except Exception:
        print("\nSUCCESS: Ingestion complete into Qdrant DB.")


if __name__ == "__main__":
    main()
