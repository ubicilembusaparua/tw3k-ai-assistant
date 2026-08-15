"""Dense retrieval, local reranking, context selection, and diagnostics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from src.config import QdrantUnavailableError, Settings
from src.dataset import TranscriptRecord, validate_record


INSUFFICIENT_EVIDENCE = "insufficient_evidence"
READY = "ready"


class QueryEmbedder(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs: object) -> Any: ...


class Reranker(Protocol):
    def predict(self, sentence_pairs: Sequence[tuple[str, str]], **kwargs: object) -> Any: ...


class SearchClient(Protocol):
    def query_points(self, collection_name: str, **kwargs: object) -> Any: ...


@dataclass(frozen=True, slots=True)
class Passage:
    chunk_ids: tuple[str, ...]
    video_id: str
    chunk_indexes: tuple[int, ...]
    text: str
    video_title: str
    channel: str
    start_time: float
    end_time: float
    formatted_time: str
    timestamp_link: str
    video_url: str
    retrieval_score: float
    rerank_score: float

    @property
    def estimated_tokens(self) -> int:
        return max(1, (len(self.text) + 3) // 4)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    status: str
    question: str
    passages: tuple[Passage, ...]
    reason: str | None
    candidate_count: int

    @property
    def has_sufficient_evidence(self) -> bool:
        return self.status == READY

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "question": self.question,
            "reason": self.reason,
            "candidate_count": self.candidate_count,
            "passages": [asdict(passage) for passage in self.passages],
        }


def _default_client(settings: Settings) -> SearchClient:
    from qdrant_client import QdrantClient

    return QdrantClient(url=settings.qdrant_url, timeout=10)


def _default_embedder(settings: Settings) -> QueryEmbedder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def _default_reranker(settings: Settings) -> Reranker:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model)


def _one_vector(encoded: Any, dimensions: int) -> list[float]:
    raw = encoded.tolist() if hasattr(encoded, "tolist") else encoded
    if len(raw) != 1:
        raise ValueError(f"question embedder returned {len(raw)} vectors, expected one")
    vector = list(raw[0])
    if len(vector) != dimensions:
        raise ValueError(
            f"question embedder returned {len(vector)} dimensions, expected {dimensions}"
        )
    return vector


def _scores(values: Any, expected: int) -> list[float]:
    raw = values.tolist() if hasattr(values, "tolist") else values
    scores = [float(value) for value in raw]
    if len(scores) != expected:
        raise ValueError(f"reranker returned {len(scores)} scores for {expected} passages")
    return scores


def _passage(record: TranscriptRecord, retrieval_score: float, rerank_score: float) -> Passage:
    data = record.data
    return Passage(
        chunk_ids=(record.chunk_id,),
        video_id=record.video_id,
        chunk_indexes=(record.chunk_index,),
        text=record.text,
        video_title=str(data["video_title"]),
        channel=str(data["channel"]),
        start_time=float(data["start_time"]),
        end_time=float(data["end_time"]),
        formatted_time=str(data["formatted_time"]),
        timestamp_link=record.timestamp_link,
        video_url=str(data["video_url"]),
        retrieval_score=retrieval_score,
        rerank_score=rerank_score,
    )


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def overlap_ratio(left: str, right: str) -> float:
    """Return containment overlap so a short duplicate cannot dominate context."""

    left_tokens, right_tokens = _tokens(left), _tokens(right)
    smaller = min(len(left_tokens), len(right_tokens))
    if not smaller:
        return 0.0
    return len(left_tokens & right_tokens) / smaller


def _append_without_overlap(left: str, right: str) -> str:
    left_words, right_words = left.split(), right.split()
    max_overlap = min(len(left_words), len(right_words), 80)
    for size in range(max_overlap, 4, -1):
        if [word.casefold() for word in left_words[-size:]] == [
            word.casefold() for word in right_words[:size]
        ]:
            return " ".join(left_words + right_words[size:])
    return f"{left.rstrip()} {right.lstrip()}"


def _merge_adjacent(left: Passage, right: Passage) -> Passage:
    ordered = sorted((left, right), key=lambda passage: min(passage.chunk_indexes))
    first, second = ordered
    indexes = tuple(sorted(first.chunk_indexes + second.chunk_indexes))
    return Passage(
        chunk_ids=first.chunk_ids + second.chunk_ids,
        video_id=first.video_id,
        chunk_indexes=indexes,
        text=_append_without_overlap(first.text, second.text),
        video_title=first.video_title,
        channel=first.channel,
        start_time=min(first.start_time, second.start_time),
        end_time=max(first.end_time, second.end_time),
        formatted_time=f"{first.formatted_time.split(' - ', maxsplit=1)[0]} - "
        f"{second.formatted_time.rsplit(' - ', maxsplit=1)[-1]}",
        timestamp_link=first.timestamp_link,
        video_url=first.video_url,
        retrieval_score=max(first.retrieval_score, second.retrieval_score),
        rerank_score=max(first.rerank_score, second.rerank_score),
    )


def select_context(candidates: Sequence[Passage], settings: Settings) -> tuple[Passage, ...]:
    selected: list[Passage] = []
    used_tokens = 0
    for candidate in sorted(candidates, key=lambda item: item.rerank_score, reverse=True):
        if candidate.rerank_score < settings.relevance_threshold:
            continue

        merged = False
        for index, existing in enumerate(selected):
            same_video = candidate.video_id == existing.video_id
            candidate_index = candidate.chunk_indexes[0]
            adjacent = same_video and any(
                abs(candidate_index - existing_index) == 1
                for existing_index in existing.chunk_indexes
            )
            if (
                adjacent
                and len(existing.chunk_indexes) <= settings.neighbor_chunk_expansion
            ):
                combined = _merge_adjacent(existing, candidate)
                new_total = used_tokens - existing.estimated_tokens + combined.estimated_tokens
                if new_total <= settings.context_token_budget:
                    selected[index] = combined
                    used_tokens = new_total
                merged = True
                break
            if (
                same_video
                and overlap_ratio(candidate.text, existing.text)
                >= settings.overlap_threshold
            ):
                merged = True
                break
        if merged:
            continue

        if used_tokens + candidate.estimated_tokens > settings.context_token_budget:
            continue
        selected.append(candidate)
        used_tokens += candidate.estimated_tokens
        if len(selected) >= settings.final_context_count:
            break

    return tuple(selected)


def retrieve(
    question: str,
    settings: Settings,
    *,
    client_factory: Callable[[Settings], SearchClient] = _default_client,
    embedder_factory: Callable[[Settings], QueryEmbedder] = _default_embedder,
    reranker_factory: Callable[[Settings], Reranker] = _default_reranker,
) -> RetrievalResult:
    normalized_question = question.strip()
    if not normalized_question:
        return RetrievalResult(
            INSUFFICIENT_EVIDENCE, question, (), "question is empty", 0
        )

    embedder = embedder_factory(settings)
    vector = _one_vector(
        embedder.encode(
            [normalized_question], show_progress_bar=False, normalize_embeddings=True
        ),
        settings.embedding_dimensions,
    )
    try:
        response = client_factory(settings).query_points(
            collection_name=settings.qdrant_collection,
            query=vector,
            limit=settings.retrieval_candidate_count,
            with_payload=True,
        )
    except Exception as error:
        raise QdrantUnavailableError(
            f"Qdrant retrieval failed at {settings.qdrant_url}: {error}"
        ) from error

    points = list(getattr(response, "points", response))
    validated: list[tuple[TranscriptRecord, float]] = []
    for point in points:
        try:
            validated.append((validate_record(point.payload), float(point.score)))
        except (AttributeError, TypeError, ValueError):
            continue

    if not validated:
        return RetrievalResult(
            INSUFFICIENT_EVIDENCE,
            normalized_question,
            (),
            "no valid transcript candidates were returned",
            len(points),
        )

    reranker = reranker_factory(settings)
    rerank_scores = _scores(
        reranker.predict(
            [
                (
                    normalized_question,
                    f"{record.data['video_title']}\n{record.text}",
                )
                for record, _ in validated
            ],
            show_progress_bar=False,
        ),
        len(validated),
    )
    ranked = [
        _passage(record, retrieval_score, rerank_score)
        for (record, retrieval_score), rerank_score in zip(
            validated, rerank_scores, strict=True
        )
    ]
    context = select_context(ranked, settings)
    if not context:
        best_score = max(rerank_scores)
        return RetrievalResult(
            INSUFFICIENT_EVIDENCE,
            normalized_question,
            (),
            f"best reranking score {best_score:.3f} is below relevance threshold "
            f"{settings.relevance_threshold:.3f}",
            len(points),
        )

    return RetrievalResult(READY, normalized_question, context, None, len(points))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="question to retrieve transcript evidence for")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = retrieve(args.question, Settings.from_env())
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0 if result.has_sufficient_evidence else 1


if __name__ == "__main__":
    raise SystemExit(main())
