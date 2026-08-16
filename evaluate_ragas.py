"""Fast Ragas Evaluation Script reading human-evaluated ground data (results/eval_dataset.csv or .json) and saving results/search_evals.csv."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List
from src.dataset import load_dataset
from src.bm25_retriever import BM25Retriever
from src.qdrant_retriever import QdrantRetriever
from src.hybrid_retriever import HybridRetriever
from src.evaluation import RagasEvaluator, save_summary_csv


def load_eval_benchmark() -> List[Dict[str, Any]]:
    """Loads benchmark evaluation samples from results/eval_dataset.csv or results/eval_dataset.json."""
    csv_file = Path("results/eval_dataset.csv")
    json_file = Path("results/eval_dataset.json")

    if csv_file.exists():
        print(f"Loading human-evaluated benchmark queries from {csv_file.resolve()}...")
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    elif json_file.exists():
        print(f"Loading benchmark queries from {json_file.resolve()}...")
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print("Benchmark file not found. Generating default 35-query dataset...")
        from generate_eval_dataset import main as gen_main
        gen_main()
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)


def main():
    print("==================================================")
    print(" Ragas Evaluation - Ground Data Benchmark Suite")
    print("==================================================\n")

    # 1. Load transcript dataset for retrievers
    chunks = load_dataset("tw3k_dataset.jsonl")
    print(f"Loaded {len(chunks)} document chunks from tw3k_dataset.jsonl.\n")

    # 2. Instantiate Retrievers
    print("Building BM25 Lexical Retriever...")
    bm25 = BM25Retriever(chunks)

    print("Connecting to Qdrant Vector Database...")
    qdrant = QdrantRetriever(collection_name="tw3k_transcripts")
    if qdrant.get_point_count() == 0:
        print(f"Qdrant collection empty. Ingesting {len(chunks)} chunks...")
        qdrant.index_chunks(chunks, batch_size=64)
    else:
        print(f"Connected to Qdrant DB: {qdrant.get_point_count()} existing points found.")

    print("Building Hybrid RRF Retriever...")
    hybrid = HybridRetriever(bm25, qdrant, rrf_k=60)

    # 3. Load human-evaluated benchmark dataset
    eval_queries = load_eval_benchmark()
    print(f"Successfully loaded {len(eval_queries)} benchmark queries for evaluation.")

    # 4. Collect top-10 retrieved contexts for each query across all 3 retrievers
    top_k_eval = 10
    bm25_samples = []
    qdrant_samples = []
    hybrid_samples = []

    print(f"\nExecuting search queries across all retrievers (top_k={top_k_eval})...")
    for item in eval_queries:
        q = item["question"]
        gt = item["ground_truth"]

        bm25_samples.append({
            "question": q,
            "ground_truth": gt,
            "retrieved_results": bm25.search(q, top_k=top_k_eval),
        })

        qdrant_samples.append({
            "question": q,
            "ground_truth": gt,
            "retrieved_results": qdrant.search(q, top_k=top_k_eval),
        })

        hybrid_samples.append({
            "question": q,
            "ground_truth": gt,
            "retrieved_results": hybrid.search(q, top_k=top_k_eval),
        })

    # 5. Evaluate using RagasEvaluator
    print("Calculating Context Precision & Context Recall metrics...")
    evaluator = RagasEvaluator()

    bm25_metrics = evaluator.evaluate_retriever(bm25_samples, retriever_name="BM25 Lexical")
    qdrant_metrics = evaluator.evaluate_retriever(qdrant_samples, retriever_name="Qdrant Vector DB")
    hybrid_metrics = evaluator.evaluate_retriever(hybrid_samples, retriever_name="Hybrid RRF")

    all_metrics = [bm25_metrics, qdrant_metrics, hybrid_metrics]

    print("\n==================================================")
    print(f" BENCHMARK SUMMARY METRICS ({len(eval_queries)} Queries @ Top-10)")
    print("==================================================")
    
    for eval_res in all_metrics:
        r_name = eval_res["retriever"]
        scores = eval_res["scores"]
        prec = scores.get("context_precision", 0.0)
        rec = scores.get("context_recall", 0.0)
        print(f"\n[{r_name}]")
        print(f"  - Context Precision @10 : {prec:.4f}")
        print(f"  - Context Recall @10    : {rec:.4f}")

    # 6. Save summary metrics into results/search_evals.csv
    csv_path = save_summary_csv(all_metrics, output_path="results/search_evals.csv")
    print(f"\nSUCCESS: Summary metrics saved to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
