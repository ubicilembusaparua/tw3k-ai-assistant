"""Ragas Evaluation Script using full tw3k_dataset.jsonl corpus."""

from src.dataset import load_dataset
from src.bm25_retriever import BM25Retriever
from src.qdrant_retriever import QdrantRetriever
from src.hybrid_retriever import HybridRetriever
from src.evaluation import RagasEvaluator


def main():
    print("==================================================")
    print(" Ragas Evaluation on tw3k_dataset.jsonl Corpus")
    print("==================================================\n")

    # 1. Load full transcript dataset
    chunks = load_dataset("tw3k_dataset.jsonl")
    print(f"Loaded {len(chunks)} document chunks from tw3k_dataset.jsonl.\n")

    # 2. Instantiate Retrievers
    print("Building BM25 Lexical Retriever...")
    bm25 = BM25Retriever(chunks)

    print("Building Qdrant Vector Retriever (ONNX Embedder)...")
    qdrant = QdrantRetriever(collection_name="ragas_tw3k_full", in_memory=True)
    qdrant.index_chunks(chunks, batch_size=64)

    print("Building Hybrid RRF Retriever...")
    hybrid = HybridRetriever(bm25, qdrant, rrf_k=60)

    # 3. Evaluation dataset queries with ground truths
    eval_queries = [
        {
            "question": "How to manage public order and reduce corruption in commanderies?",
            "ground_truth": "Build a Grand Inspectorate, lower tax rates, and assign high Authority administrators to maintain public order.",
        },
        {
            "question": "What is Cao Cao's diplomatic proxy war mechanic?",
            "ground_truth": "Cao Cao uses Credibility points to manipulate diplomatic relations and incite proxy wars between warlords.",
        },
        {
            "question": "How to command armies and generals in battle?",
            "ground_truth": "Armies are composed of up to three generals each commanding up to six unit retinues matched to their element.",
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
            "retrieved_results": bm25.search(q, top_k=3),
        })

        qdrant_samples.append({
            "question": q,
            "ground_truth": gt,
            "retrieved_results": qdrant.search(q, top_k=3),
        })

        hybrid_samples.append({
            "question": q,
            "ground_truth": gt,
            "retrieved_results": hybrid.search(q, top_k=3),
        })

    # 5. Evaluate using RagasEvaluator
    evaluator = RagasEvaluator()

    bm25_metrics = evaluator.evaluate_retriever(bm25_samples, retriever_name="BM25 Lexical")
    qdrant_metrics = evaluator.evaluate_retriever(qdrant_samples, retriever_name="Qdrant Vector DB")
    hybrid_metrics = evaluator.evaluate_retriever(hybrid_samples, retriever_name="Hybrid RRF")

    print("\n==================================================")
    print(" RAGAS EVALUATION METRICS COMPARISON (TW3K Corpus) ")
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
