import os
from typing import Any, Dict, List, Optional
from datasets import Dataset
from src.schema import SearchResult


class RagasEvaluator:
    """Evaluates retrieval quality (Context Precision & Context Recall) using Ragas."""

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
        """Runs evaluation on formatted dataset samples.
        
        Calculates Context Precision & Context Recall scores.
        """
        dataset = self.format_dataset(eval_samples)

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
                    "status": "success",
                }
            except Exception as e:
                # Fallback heuristics if API call fails
                return self._fallback_evaluate(dataset, retriever_name, error_msg=str(e))
        else:
            return self._fallback_evaluate(dataset, retriever_name, error_msg="OPENAI_API_KEY not set")

    def _fallback_evaluate(
        self,
        dataset: Dataset,
        retriever_name: str,
        error_msg: str,
    ) -> Dict[str, Any]:
        """Calculates deterministic text overlap context metrics when offline."""
        precision_scores = []
        recall_scores = []

        for item in dataset:
            contexts = item["contexts"]
            gt = item["ground_truth"].lower()

            if not gt or not contexts:
                precision_scores.append(0.0)
                recall_scores.append(0.0)
                continue

            # Heuristic: Check fraction of ground truth keywords found in top retrieved context
            gt_words = set(w for w in gt.split() if len(w) > 3)
            if not gt_words:
                precision_scores.append(1.0)
                recall_scores.append(1.0)
                continue

            # Top context precision
            top_ctx = contexts[0].lower() if contexts else ""
            found_top = sum(1 for w in gt_words if w in top_ctx)
            precision = found_top / len(gt_words)

            # Overall recall across all top-k contexts
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
            "status": "offline_heuristic",
            "info": error_msg,
        }
