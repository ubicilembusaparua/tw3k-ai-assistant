import os
from typing import Any, Dict, List, Optional
from datasets import Dataset
from src.schema import SearchResult


class RagasEvaluator:
    """Evaluates retrieval quality (Context Precision & Context Recall) per retrieved context."""

    def __init__(self, metrics: Optional[List[Any]] = None):
        self.metrics = metrics

    def format_dataset(
        self,
        eval_samples: List[Dict[str, Any]],
    ) -> Dataset:
        """Formats query samples and retrieved search results into a Ragas Dataset.

        Expected sample structure:
        {
            "question": str,
            "retrieved_results": List[SearchResult],
            "ground_truth": str or List[str]
        }
        """
        data = {
            "question": [],
            "contexts": [],
            "ground_truth": [],
        }

        for sample in eval_samples:
            question = sample["question"]
            results: List[SearchResult] = sample["retrieved_results"]
            gt = sample.get("ground_truth", "")
            if isinstance(gt, list):
                gt = " ".join(gt)

            contexts = [res.chunk.content for res in results]

            data["question"].append(question)
            data["contexts"].append(contexts)
            data["ground_truth"].append(gt)

        return Dataset.from_dict(data)

    def evaluate_retriever(
        self,
        eval_samples: List[Dict[str, Any]],
        retriever_name: str = "Retriever",
    ) -> Dict[str, Any]:
        """Runs evaluation on formatted dataset samples, evaluating each retrieved context individually."""
        dataset = self.format_dataset(eval_samples)

        # Calculate detailed per-context evaluation for each retrieved item across samples
        per_sample_details = []
        for sample in eval_samples:
            q = sample["question"]
            gt = sample.get("ground_truth", "")
            results: List[SearchResult] = sample.get("retrieved_results", [])
            gt_words = set(w.lower() for w in (gt if isinstance(gt, str) else " ".join(gt)).split() if len(w) > 3)

            context_evals = []
            for res in results:
                ctx_lower = res.chunk.content.lower()
                matches = [w for w in gt_words if w in ctx_lower] if gt_words else []
                relevance_score = len(matches) / max(len(gt_words), 1) if gt_words else 1.0
                context_evals.append({
                    "rank": res.rank,
                    "chunk_id": res.chunk.id,
                    "retriever_score": res.score,
                    "relevance_score": round(relevance_score, 4),
                    "matched_keywords": matches,
                    "snippet": res.chunk.content[:120].replace("\n", " "),
                })

            per_sample_details.append({
                "question": q,
                "ground_truth": gt,
                "context_evaluations": context_evals,
            })

        # Check if OpenAI API key or custom LLM endpoint is set for Ragas execution
        if os.getenv("OPENAI_API_KEY"):
            try:
                from ragas import evaluate
                from ragas.metrics import context_precision, context_recall

                metrics = self.metrics or [context_precision, context_recall]
                results = evaluate(dataset=dataset, metrics=metrics)
                return {
                    "retriever": retriever_name,
                    "scores": dict(results),
                    "sample_details": per_sample_details,
                    "status": "success",
                }
            except Exception as e:
                return self._fallback_evaluate(dataset, retriever_name, per_sample_details, error_msg=str(e))
        else:
            return self._fallback_evaluate(dataset, retriever_name, per_sample_details, error_msg="OPENAI_API_KEY not set")

    def _fallback_evaluate(
        self,
        dataset: Dataset,
        retriever_name: str,
        per_sample_details: List[Dict[str, Any]],
        error_msg: str,
    ) -> Dict[str, Any]:
        """Calculates deterministic context precision and recall scores per query and top-K context list."""
        precision_scores = []
        recall_scores = []

        for item in dataset:
            contexts = item["contexts"]
            gt = item["ground_truth"].lower()

            if not gt or not contexts:
                precision_scores.append(0.0)
                recall_scores.append(0.0)
                continue

            gt_words = set(w for w in gt.split() if len(w) > 3)
            if not gt_words:
                precision_scores.append(1.0)
                recall_scores.append(1.0)
                continue

            top_ctx = contexts[0].lower() if contexts else ""
            found_top = sum(1 for w in gt_words if w in top_ctx)
            precision = found_top / len(gt_words)

            all_ctx = " ".join(contexts).lower()
            found_all = sum(1 for w in gt_words if w in all_ctx)
            recall = found_all / len(gt_words)

            precision_scores.append(precision)
            recall_scores.append(recall)

        avg_precision = sum(precision_scores) / max(len(precision_scores), 1)
        avg_recall = sum(recall_scores) / max(len(recall_scores), 1)

        return {
            "retriever": retriever_name,
            "scores": {
                "context_precision": round(avg_precision, 4),
                "context_recall": round(avg_recall, 4),
            },
            "sample_details": per_sample_details,
            "status": "offline_heuristic",
            "info": error_msg,
        }
