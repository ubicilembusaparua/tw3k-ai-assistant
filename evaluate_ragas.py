"""Fast Ragas Evaluation Script pulling existing vectors from Qdrant DB and saving summary metrics to results/search_evals.csv."""

from src.dataset import load_dataset
from src.bm25_retriever import BM25Retriever
from src.qdrant_retriever import QdrantRetriever
from src.hybrid_retriever import HybridRetriever
from src.evaluation import RagasEvaluator, save_summary_csv


def main():
    print("==================================================")
    print(" Ragas Top-10 Retrieval Evaluation (tw3k_dataset.jsonl)")
    print("==================================================\n")

    # 1. Load full transcript dataset for BM25
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
    ]

    # 4. Collect top-10 retrieved contexts for each query across all 3 retrievers
    top_k_eval = 10
    bm25_samples = []
    qdrant_samples = []
    hybrid_samples = []

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
    evaluator = RagasEvaluator()

    bm25_metrics = evaluator.evaluate_retriever(bm25_samples, retriever_name="BM25 Lexical")
    qdrant_metrics = evaluator.evaluate_retriever(qdrant_samples, retriever_name="Qdrant Vector DB")
    hybrid_metrics = evaluator.evaluate_retriever(hybrid_samples, retriever_name="Hybrid RRF")

    all_metrics = [bm25_metrics, qdrant_metrics, hybrid_metrics]

    print("\n==================================================")
    print(" SUMMARY METRICS (Top-10 Retrieval List) ")
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

    print("\n==================================================")
    print(" INDIVIDUAL PER-CONTEXT EVALUATIONS (Top-10 Ranks) ")
    print("==================================================")

    for eval_res in all_metrics:
        r_name = eval_res["retriever"]
        print(f"\n==================================================")
        print(f" RETRIEVER: {r_name}")
        print(f"==================================================")

        for s_idx, sample in enumerate(eval_res["sample_details"], start=1):
            print(f"\nQuery #{s_idx}: '{sample['question']}'")
            print(f"Ground Truth: '{sample['ground_truth']}'")
            print("-" * 75)

            for ctx_eval in sample["context_evaluations"]:
                print(
                    f"  [Rank {ctx_eval['rank']:2d}] Score: {ctx_eval['retriever_score']:7.4f} | "
                    f"Relevance: {ctx_eval['relevance_score']:6.2f} | "
                    f"Matches: {ctx_eval['matched_keywords']}"
                )
                print(f"            Snippet: \"{ctx_eval['snippet']}...\"")


if __name__ == "__main__":
    main()
