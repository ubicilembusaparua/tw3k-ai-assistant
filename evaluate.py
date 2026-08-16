"""Interactive evaluation script comparing BM25, Qdrant Vector, and Hybrid (RRF) retrieval strategies side-by-side."""

from src.schema import DocumentChunk
from src.bm25_retriever import BM25Retriever
from src.qdrant_retriever import QdrantRetriever
from src.hybrid_retriever import HybridRetriever


def main():
    print("==================================================")
    print(" RAG Retrieval Comparison: BM25 vs Qdrant vs Hybrid")
    print("==================================================\n")

    # 1. Corpus of strategy & domain knowledge passages
    corpus = [
        DocumentChunk(
            id="doc_1",
            content="To maintain public order in a high-corruption commandery, build a Grand Inspectorate and assign administrators with high Authority stats.",
        ),
        DocumentChunk(
            id="doc_2",
            content="Boosting agricultural output requires upgrading farmland buildings, constructing state workshops, and researching tax reforms in the reform tree.",
        ),
        DocumentChunk(
            id="doc_3",
            content="When fighting against superior enemy numbers, use bottleneck choke points in mountain passes and deploy heavy spear infantry to hold the line.",
        ),
        DocumentChunk(
            id="doc_4",
            content="Cao Cao's unique faction mechanic is Credibility, allowing him to manipulate diplomatic relations and incite proxy wars between rival warlords.",
        ),
        DocumentChunk(
            id="doc_5",
            content="Managing army upkeep costs: disband unnecessary militia regiments during peacetime and stack corruption reduction traits on court ministers.",
        ),
    ]

    print(f"Indexed Corpus Size: {len(corpus)} passages.\n")

    # 2. Initialize Retrievers
    print("Initializing BM25 Lexical Retriever...")
    bm25_retriever = BM25Retriever(corpus)

    print("Initializing Qdrant Vector Retriever...")
    qdrant_retriever = QdrantRetriever(collection_name="evaluate_corpus", in_memory=True)
    qdrant_retriever.index_chunks(corpus)

    print("Initializing Hybrid Retriever (RRF Fusion)...")
    hybrid_retriever = HybridRetriever(bm25_retriever, qdrant_retriever, rrf_k=60)

    print("\n--------------------------------------------------")

    # 3. Test Queries demonstrating different retrieval challenges
    test_queries = [
        ("Exact Keyword Search", "Grand Inspectorate corruption authority"),
        ("Conceptual / Synonym Search", "How to manage money and reduce military expenses during peacetime?"),
        ("Hybrid Mixed Query", "Cao Cao proxy wars and diplomatic credibility mechanics"),
    ]

    for label, query in test_queries:
        print(f"\nQUERY [{label}]: '{query}'")
        print("=" * 65)

        bm25_res = bm25_retriever.search(query, top_k=2)
        qdrant_res = qdrant_retriever.search(query, top_k=2)
        hyb_res = hybrid_retriever.search(query, top_k=2)

        def fmt(res):
            return f"[{res[0].chunk.id}] {res[0].chunk.content[:65]}... (Score: {res[0].score:.4f})" if res else "N/A"

        print(f"  [BM25   Top-1]: {fmt(bm25_res)}")
        print(f"  [Qdrant Top-1]: {fmt(qdrant_res)}")
        print(f"  [Hybrid Top-1]: {fmt(hyb_res)}")


if __name__ == "__main__":
    main()
