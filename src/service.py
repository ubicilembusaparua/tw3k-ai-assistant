"""Application orchestration and cached local model lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any

from src.config import QdrantUnavailableError, Settings
from src.generation import GroundedAnswer, OpenAIClient, generate_answer
from src.retrieval import QueryEmbedder, Reranker, SearchClient, retrieve


class EmptyIndexError(RuntimeError):
    """Raised when Qdrant is available but contains no transcript points."""


class DuplicateSubmissionError(RuntimeError):
    """Raised when an identical question is already being processed."""


def _client(settings: Settings) -> SearchClient:
    from qdrant_client import QdrantClient

    return QdrantClient(url=settings.qdrant_url, timeout=10)


def _embedder(settings: Settings) -> QueryEmbedder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def _reranker(settings: Settings) -> Reranker:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model)


class AssistantService:
    """Own cached clients/models and orchestrate retrieval before generation."""

    def __init__(
        self,
        settings: Settings,
        *,
        client_factory: Callable[[Settings], SearchClient] = _client,
        embedder_factory: Callable[[Settings], QueryEmbedder] = _embedder,
        reranker_factory: Callable[[Settings], Reranker] = _reranker,
        openai_client_factory: Callable[[Settings], OpenAIClient] | None = None,
    ) -> None:
        self.settings = settings
        self._client_factory = client_factory
        self._embedder_factory = embedder_factory
        self._reranker_factory = reranker_factory
        self._openai_client_factory = openai_client_factory
        self._client: SearchClient | None = None
        self._embedder: QueryEmbedder | None = None
        self._reranker: Reranker | None = None
        self._load_lock = Lock()
        self._question_lock = Lock()
        self._active_questions: set[str] = set()

    def _get_client(self) -> SearchClient:
        if self._client is None:
            with self._load_lock:
                if self._client is None:
                    self._client = self._client_factory(self.settings)
        return self._client

    def _load_models(self) -> tuple[QueryEmbedder, Reranker]:
        if self._embedder is None or self._reranker is None:
            with self._load_lock:
                if self._embedder is None:
                    self._embedder = self._embedder_factory(self.settings)
                if self._reranker is None:
                    self._reranker = self._reranker_factory(self.settings)
        return self._embedder, self._reranker

    def _point_count(self, client: SearchClient) -> int:
        try:
            collection_exists = getattr(client, "collection_exists", None)
            if callable(collection_exists) and not collection_exists(
                self.settings.qdrant_collection
            ):
                return 0
            result = client.count(
                collection_name=self.settings.qdrant_collection, exact=True
            )  # type: ignore[attr-defined]
            return int(result.count)
        except Exception as error:
            raise QdrantUnavailableError(
                f"Qdrant index check failed at {self.settings.qdrant_url}: {error}"
            ) from error

    def readiness(self) -> dict[str, object]:
        qdrant_ready = False
        index_count: int | None = None
        qdrant_error: str | None = None
        try:
            client = self._get_client()
            client.get_collections()  # type: ignore[attr-defined]
            index_count = self._point_count(client)
            qdrant_ready = True
        except Exception as error:
            qdrant_error = str(error)

        components = {
            "app": {"ready": True},
            "qdrant": {
                "ready": qdrant_ready,
                "index_count": index_count,
                "error": qdrant_error,
            },
            "embedding_model": {"ready": self._embedder is not None},
            "reranker_model": {"ready": self._reranker is not None},
            "openai": {
                "ready": bool(self.settings.openai_api_key),
                "model": self.settings.openai_model,
            },
        }
        ready = all(
            (
                qdrant_ready,
                bool(index_count),
                self._embedder is not None,
                self._reranker is not None,
                bool(self.settings.openai_api_key),
            )
        )
        return {"ready": ready, "components": components}

    def ask(self, question: str) -> GroundedAnswer:
        normalized = " ".join(question.split())
        with self._question_lock:
            if normalized in self._active_questions:
                raise DuplicateSubmissionError(
                    "this question is already being processed"
                )
            self._active_questions.add(normalized)

        try:
            client = self._get_client()
            if self._point_count(client) == 0:
                raise EmptyIndexError(
                    "the transcript index is empty; run tw3k-index before asking questions"
                )
            embedder, reranker = self._load_models()
            evidence = retrieve(
                normalized,
                self.settings,
                client_factory=lambda _: client,
                embedder_factory=lambda _: embedder,
                reranker_factory=lambda _: reranker,
            )
            if self._openai_client_factory is None:
                return generate_answer(normalized, evidence, self.settings)
            return generate_answer(
                normalized,
                evidence,
                self.settings,
                client_factory=self._openai_client_factory,
            )
        finally:
            with self._question_lock:
                self._active_questions.discard(normalized)
