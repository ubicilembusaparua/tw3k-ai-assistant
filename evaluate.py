"""Side-by-side RAG Retrieval Evaluation on the full tw3k_dataset.jsonl transcript corpus."""

from src.dataset import load_dataset
from src.bm25_retriever import BM25Retriever
from src.qdrant_retriever import QdrantRetriever
from src.hybrid_retriever import HybridRetriever


def main():
    print("==================================================")
    print(" Full Dataset Retrieval Evaluation (tw3k_dataset.jsonl)")
    print("==================================================\n")

    # 1. Load full transcript dataset
    print("Loading transcript chunks from tw3k_dataset.jsonl...")
    chunks = load_dataset("tw3k_dataset.jsonl")
    print(f"Loaded {len(chunks)} transcript chunks.\n")

    # 2. Initialize BM25 Lexical Retriever directly from loaded dataset
    print("Building BM25 Lexical index...")
    bm25_retriever = BM25Retriever(chunks)

    # 3. Initialize Qdrant Vector Retriever and index embeddings
    print("Initializing Qdrant Vector Database (ONNX Embedder)...")
    qdrant_retriever = QdrantRetriever(collection_name="tw3k_transcripts_full", in_memory=True)
    print(f"Upserting {len(chunks)} vectors into Qdrant DB...")
    qdrant_retriever.index_chunks(chunks, batch_size=64)

    # 4. Initialize Hybrid RRF Retriever (Best of BM25 + Qdrant DB)
    print("Initializing Hybrid RRF Fusion Retriever...")
    hybrid_retriever = HybridRetriever(bm25_retriever, qdrant_retriever, rrf_k=60)

    print("\n--------------------------------------------------")

    # 5. Real Total War: Three Kingdoms queries
    queries = [
        "How to manage public order and reduce corruption in commanderies?",
        "Cao Cao credibility mechanics and diplomatic proxy wars",
        "Best cavalry flank tactics and spear infantry line defense",
    ]

    for query in queries:
        print(f"\nQUERY: '{query}'")
        print("=" * 70)

        bm25_res = bm25_retriever.search(query, top_k=1)
        qdrant_res = qdrant_retriever.search(query, top_k=1)
        hybrid_res = hybrid_retriever.search(query, top_k=1)

        def print_result(retriever_name, res):
            if not res:
                print(f"  [{retriever_name}]: No result")
                return
            item = res[0]
            title = item.chunk.metadata.get("video_title", "Unknown")
            time_str = item.chunk.metadata.get("formatted_time", "")
            snippet = item.chunk.content[:120].replace("\n", " ")
            print(f"  [{retriever_name:12s}] Score: {item.score:7.4f} | ID: {item.chunk.id}")
            print(f"                 Video: {title} ({time_str})")
            print(f"                 Snippet: \"{snippet}...\"")

        print_result("BM25", bm25_res)
        print_result("Qdrant DB", qdrant_res)
        print_result("Hybrid RRF", hybrid_res)


if __name__ == "__main__":
    main()
