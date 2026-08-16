"""Ragas Evaluation Script comparing BM25, Qdrant Vector (ONNX), and Hybrid (RRF) Retrieval Quality."""

from src.schema import DocumentChunk
from src.bm25_retriever import BM25Retriever
from src.qdrant_retriever import QdrantRetriever
from src.hybrid_retriever import HybridRetriever
from src.evaluation import RagasEvaluator


def main():
    print("==================================================")
    print(" RAG Retrieval Evaluation with Ragas & Context Metrics")
    print("==================================================\n")

    # 1. Corpus of domain knowledge passages
    corpus = [
        DocumentChunk(
            id="doc_1",
            content="To maintain public order in a high-corruption commandery, build a Grand Inspectorate and assign administrators with high Authority stats.",
            metadata={"category": "economy"},
        ),
        DocumentChunk(
            id="doc_2",
            content="Boosting agricultural output requires upgrading farmland buildings, constructing state workshops, and researching tax reforms in the reform tree.",
            metadata={"category": "economy"},
        ),
        DocumentChunk(
            id="doc_3",
            content="When fighting against superior enemy numbers, use bottleneck choke points in mountain passes and deploy heavy spear infantry to hold the line.",
            metadata={"category": "military"},
        ),
        DocumentChunk(
            id="doc_4",
            content="Cao Cao's unique faction mechanic is Credibility, allowing him to manipulate diplomatic relations and incite proxy wars between rival warlords.",
            metadata={"category": "diplomacy"},
        ),
        DocumentChunk(
            id="doc_5",
            content="Managing army upkeep costs: disband unnecessary militia regiments during peacetime and stack corruption reduction traits on court ministers.",
            metadata={"category": "military"},
        ),
    ]

    print(f"Indexed Corpus: {len(corpus)} document chunks.")

    # 2. Instantiate Retrievers
    bm25 = BM25Retriever(corpus)
    qdrant = QdrantRetriever(collection_name="ragas_eval_corpus", in_memory=True)
    qdrant.index_chunks(corpus)
    hybrid = HybridRetriever(bm25, qdrant, rrf_k=60)

    # 3. Ground-truth evaluation dataset queries
    eval_queries = [
        {
            "question": "How to lower corruption and improve public order in commanderies?",
            "ground_truth": "Build a Grand Inspectorate and assign high Authority administrators to maintain public order and reduce corruption.",
        },
        {
            "question": "What is Cao Cao's special diplomatic mechanic?",
            "ground_truth": "Cao Cao uses Credibility to incite proxy wars and manipulate diplomatic relations with rival warlords.",
        },
        {
            "question": "How to defend against larger enemy armies in battle?",
            "ground_truth": "Deploy heavy spear infantry in mountain pass bottleneck choke points to hold off superior numbers.",
        },
        {
            "question": "What are effective methods to manage peacetime military upkeep costs?",
            "ground_truth": "Disband unnecessary militia regiments and stack corruption reduction traits on court ministers.",
        },
    ]

    # 4. Collect retrieved context for each query across all 3 retrievers
    bm25_samples = []
    qdrant_samples = []
    hybrid_samples = []

    for item in eval_queries:
        q = item["question"]
        gt = item["ground_truth"]

        bm25_samples.append({
            "question": q,
            "ground_truth": gt,
            "retrieved_results": bm25.search(q, top_k=2),
        })

        qdrant_samples.append({
            "question": q,
            "ground_truth": gt,
            "retrieved_results": qdrant.search(q, top_k=2),
        })

        hybrid_samples.append({
            "question": q,
            "ground_truth": gt,
            "retrieved_results": hybrid.search(q, top_k=2),
        })

    # 5. Evaluate using RagasEvaluator
    evaluator = RagasEvaluator()

    bm25_metrics = evaluator.evaluate_retriever(bm25_samples, retriever_name="BM25 Lexical")
    qdrant_metrics = evaluator.evaluate_retriever(qdrant_samples, retriever_name="Qdrant Vector (ONNX)")
    hybrid_metrics = evaluator.evaluate_retriever(hybrid_samples, retriever_name="Hybrid RRF")

    print("\n==================================================")
    print(" RAGAS EVALUATION METRICS COMPARISON ")
    print("==================================================")
    
    for eval_res in [bm25_metrics, qdrant_metrics, hybrid_metrics]:
        r_name = eval_res["retriever"]
        scores = eval_res["scores"]
        prec = scores.get("context_precision", 0.0)
        rec = scores.get("context_recall", 0.0)
        mode = eval_res.get("status", "ok")
        print(f"\n[{r_name}] Mode: {mode}")
        print(f"  - Context Precision : {prec:.4f}")
        print(f"  - Context Recall    : {rec:.4f}")


if __name__ == "__main__":
    main()
