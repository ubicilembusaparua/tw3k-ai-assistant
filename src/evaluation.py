"""Repeatable retrieval evaluation and default-parameter tuning."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import parse_qs, urlparse

from src.config import Settings
from src.retrieval import RetrievalResult, retrieve


@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    id: str
    category: str
    question: str
    supported: bool
    expected_video_ids: tuple[str, ...]
    expected_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationProfile:
    name: str
    candidate_count: int
    final_context_count: int
    context_token_budget: int
    relevance_threshold: float
    neighbor_chunk_expansion: int
    eligible_default: bool = True

    def settings(self, base: Settings) -> Settings:
        return replace(
            base,
            retrieval_candidate_count=self.candidate_count,
            final_context_count=self.final_context_count,
            context_token_budget=self.context_token_budget,
            relevance_threshold=self.relevance_threshold,
            neighbor_chunk_expansion=self.neighbor_chunk_expansion,
        )


PROFILES = (
    EvaluationProfile("no_neighbors", 10, 6, 3000, 0.35, 0, False),
    EvaluationProfile("lean", 10, 6, 3000, 0.35, 1),
    EvaluationProfile("balanced", 20, 8, 4000, 0.35, 1),
    EvaluationProfile("strict", 20, 8, 4000, 1.0, 1),
    EvaluationProfile("broad", 30, 10, 5000, 0.0, 1),
    EvaluationProfile("moderate", 20, 8, 4000, -1.0, 1),
    EvaluationProfile("moderate_broad", 30, 10, 5000, -1.0, 1),
    EvaluationProfile("permissive", 20, 8, 4000, -2.0, 1),
    EvaluationProfile("very_permissive", 30, 10, 5000, -5.0, 1),
)


def load_questions(path: str | Path) -> tuple[EvaluationQuestion, ...]:
    questions: list[EvaluationQuestion] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            try:
                data = json.loads(raw_line)
                question = EvaluationQuestion(
                    id=str(data["id"]),
                    category=str(data["category"]),
                    question=str(data["question"]),
                    supported=bool(data["supported"]),
                    expected_video_ids=tuple(data["expected_video_ids"]),
                    expected_terms=tuple(data["expected_terms"]),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid evaluation question on line {line_number}: {error}") from error
            if not question.id or question.id in seen_ids:
                raise ValueError(f"duplicate or empty evaluation ID on line {line_number}")
            if not question.question.strip():
                raise ValueError(f"empty evaluation question on line {line_number}")
            if question.supported and not question.expected_video_ids:
                raise ValueError(
                    f"supported evaluation question {question.id} needs expected_video_ids"
                )
            seen_ids.add(question.id)
            questions.append(question)
    return tuple(questions)


def _valid_timestamp(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
        and "t" in parse_qs(parsed.query)
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def evaluate_profile(
    questions: Sequence[EvaluationQuestion],
    settings: Settings,
    retrieve_fn: Callable[[str, Settings], RetrievalResult],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    latencies: list[float] = []
    total_citations = 0
    valid_citations = 0

    for question in questions:
        started = perf_counter()
        result = retrieve_fn(question.question, settings)
        latency_ms = (perf_counter() - started) * 1000
        latencies.append(latency_ms)

        video_ids = {passage.video_id for passage in result.passages}
        combined_text = " ".join(passage.text.casefold() for passage in result.passages)
        video_hit = bool(video_ids & set(question.expected_video_ids))
        term_hit = not question.expected_terms or any(
            term.casefold() in combined_text for term in question.expected_terms
        )
        if question.supported:
            retrieval_hit = result.has_sufficient_evidence and (video_hit or term_hit)
            grounded = retrieval_hit and term_hit
            unsupported_rejected = None
        else:
            retrieval_hit = None
            grounded = None
            unsupported_rejected = not result.has_sufficient_evidence

        links = [passage.timestamp_link for passage in result.passages]
        link_checks = [_valid_timestamp(link) for link in links]
        total_citations += len(link_checks)
        valid_citations += sum(link_checks)
        records.append(
            {
                "id": question.id,
                "category": question.category,
                "supported": question.supported,
                "status": result.status,
                "retrieval_hit": retrieval_hit,
                "grounded": grounded,
                "unsupported_rejected": unsupported_rejected,
                "citation_valid": all(link_checks),
                "latency_ms": round(latency_ms, 2),
                "selected_video_ids": sorted(video_ids),
                "selected_chunk_ids": [
                    chunk_id
                    for passage in result.passages
                    for chunk_id in passage.chunk_ids
                ],
                "best_rerank_score": (
                    max((passage.rerank_score for passage in result.passages), default=None)
                ),
            }
        )

    supported = [record for record in records if record["supported"]]
    unsupported = [record for record in records if not record["supported"]]

    def rate(items: Sequence[dict[str, Any]], field: str) -> float:
        return (
            sum(bool(item[field]) for item in items) / len(items)
            if items
            else 0.0
        )

    return {
        "settings": {
            "candidate_count": settings.retrieval_candidate_count,
            "final_context_count": settings.final_context_count,
            "context_token_budget": settings.context_token_budget,
            "relevance_threshold": settings.relevance_threshold,
            "neighbor_chunk_expansion": settings.neighbor_chunk_expansion,
            "overlap_threshold": settings.overlap_threshold,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
        },
        "metrics": {
            "question_count": len(records),
            "supported_hit_rate": round(rate(supported, "retrieval_hit"), 4),
            "unsupported_rejection_rate": round(
                rate(unsupported, "unsupported_rejected"), 4
            ),
            "citation_validity": round(
                valid_citations / total_citations if total_citations else 1.0, 4
            ),
            "groundedness": round(rate(supported, "grounded"), 4),
            "latency_ms_mean": round(statistics.fmean(latencies), 2),
            "latency_ms_p50": round(_percentile(latencies, 0.5), 2),
            "latency_ms_p95": round(_percentile(latencies, 0.95), 2),
        },
        "records": records,
    }


def _quality_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = result["metrics"]
    return (
        metrics["unsupported_rejection_rate"],
        metrics["supported_hit_rate"] + metrics["groundedness"],
        metrics["citation_validity"],
        -metrics["latency_ms_p95"],
    )


def run_evaluation(
    questions: Sequence[EvaluationQuestion],
    base_settings: Settings,
    retrieve_fn: Callable[[str, Settings], RetrievalResult],
    profiles: Sequence[EvaluationProfile] = PROFILES,
) -> dict[str, Any]:
    results = {
        profile.name: evaluate_profile(
            questions, profile.settings(base_settings), retrieve_fn
        )
        for profile in profiles
    }
    eligible_names = [profile.name for profile in profiles if profile.eligible_default]
    if not eligible_names:
        raise ValueError("at least one evaluation profile must be eligible as a default")
    selected_name = max(
        eligible_names, key=lambda name: _quality_key(results[name])
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "methodology": {
            "hit_rate": "supported question selects a curated expected video or context containing a curated answer term",
            "citation_validity": "selected source links are absolute HTTP(S) URLs with a timestamp parameter",
            "groundedness": "retrieved expected-video context also contains a curated answer term; this is a retrieval-context proxy and does not call OpenAI",
            "latency": "wall-clock dense retrieval, cross-encoder reranking, gating, and context selection",
            "selection": "among profiles that retain one-chunk neighbor expansion, preserve unsupported rejection first, then maximize hit rate + groundedness, citation validity, and latency",
        },
        "selected_profile": selected_name,
        "recommended_settings": results[selected_name]["settings"],
        "profiles": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions", type=Path, default=Path("evaluation/questions.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/results.json")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    questions = load_questions(args.questions)
    if not 30 <= len(questions) <= 50:
        raise SystemExit(
            f"evaluation set must contain 30-50 questions, found {len(questions)}"
        )

    settings = Settings.from_env()
    from qdrant_client import QdrantClient
    from sentence_transformers import CrossEncoder, SentenceTransformer

    client = QdrantClient(url=settings.qdrant_url, timeout=10)
    embedder = SentenceTransformer(settings.embedding_model)
    reranker = CrossEncoder(settings.reranker_model)

    def retrieve_cached(question: str, profile_settings: Settings) -> RetrievalResult:
        return retrieve(
            question,
            profile_settings,
            client_factory=lambda _: client,
            embedder_factory=lambda _: embedder,
            reranker_factory=lambda _: reranker,
        )

    report = run_evaluation(questions, settings, retrieve_cached)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "selected_profile": report["selected_profile"],
        "recommended_settings": report["recommended_settings"],
        "metrics": report["profiles"][report["selected_profile"]]["metrics"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
