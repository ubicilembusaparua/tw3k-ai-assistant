"""Central application configuration and service availability checks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol
from urllib.parse import urlparse


class ConfigurationError(ValueError):
    """Raised when application configuration is invalid."""


class MissingConfigurationError(ConfigurationError):
    """Raised when a feature's required configuration is absent."""


class QdrantUnavailableError(RuntimeError):
    """Raised when a configured Qdrant service cannot be reached."""


class QdrantHealthClient(Protocol):
    """The minimal Qdrant client interface needed for a health check."""

    def get_collections(self) -> object: ...


def _positive_int(
    env: Mapping[str, str], name: str, default: int, *, allow_zero: bool = False
) -> int:
    raw_value = env.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer, got {raw_value!r}") from error
    if value < 0 or (value == 0 and not allow_zero):
        requirement = "zero or greater" if allow_zero else "greater than zero"
        raise ConfigurationError(f"{name} must be {requirement}")
    return value


def _unit_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw_value = env.get(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number, got {raw_value!r}") from error
    if not 0 <= value <= 1:
        raise ConfigurationError(f"{name} must be between 0 and 1")
    return value


def _bounded_float(
    env: Mapping[str, str], name: str, default: float, minimum: float, maximum: float
) -> float:
    raw_value = env.get(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number, got {raw_value!r}") from error
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuration shared by indexing, retrieval, generation, and the API."""

    openai_api_key: str | None
    openai_model: str
    embedding_model: str
    embedding_dimensions: int
    reranker_model: str
    qdrant_url: str
    qdrant_collection: str
    retrieval_candidate_count: int
    final_context_count: int
    context_token_budget: int
    relevance_threshold: float
    neighbor_chunk_expansion: int
    overlap_threshold: float

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        values = os.environ if env is None else env
        settings = cls(
            openai_api_key=values.get("OPENAI_API_KEY", "").strip() or None,
            openai_model=values.get("OPENAI_MODEL", "gpt-5-mini").strip(),
            embedding_model=values.get(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ).strip(),
            embedding_dimensions=_positive_int(values, "EMBEDDING_DIMENSIONS", 384),
            reranker_model=values.get(
                "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"
            ).strip(),
            qdrant_url=values.get("QDRANT_URL", "http://localhost:6333").strip(),
            qdrant_collection=values.get(
                "QDRANT_COLLECTION", "tw3k_transcripts"
            ).strip(),
            retrieval_candidate_count=_positive_int(
                values, "RETRIEVAL_CANDIDATE_COUNT", 10
            ),
            final_context_count=_positive_int(values, "FINAL_CONTEXT_COUNT", 6),
            context_token_budget=_positive_int(values, "CONTEXT_TOKEN_BUDGET", 3000),
            relevance_threshold=_bounded_float(
                values, "RELEVANCE_THRESHOLD", 0.35, -20.0, 20.0
            ),
            neighbor_chunk_expansion=_positive_int(
                values, "NEIGHBOR_CHUNK_EXPANSION", 1, allow_zero=True
            ),
            overlap_threshold=_unit_float(values, "OVERLAP_THRESHOLD", 0.75),
        )
        settings._validate()
        return settings

    def _validate(self) -> None:
        required_names = {
            "OPENAI_MODEL": self.openai_model,
            "EMBEDDING_MODEL": self.embedding_model,
            "RERANKER_MODEL": self.reranker_model,
            "QDRANT_COLLECTION": self.qdrant_collection,
        }
        for name, value in required_names.items():
            if not value:
                raise MissingConfigurationError(f"{name} must not be empty")

        parsed_url = urlparse(self.qdrant_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ConfigurationError(
                "QDRANT_URL must be an absolute http:// or https:// URL"
            )
        if self.final_context_count > self.retrieval_candidate_count:
            raise ConfigurationError(
                "FINAL_CONTEXT_COUNT cannot exceed RETRIEVAL_CANDIDATE_COUNT"
            )

    def require_openai_api_key(self) -> str:
        """Return the generation credential or raise a feature-specific error."""

        if not self.openai_api_key:
            raise MissingConfigurationError(
                "OPENAI_API_KEY is required for answer generation; "
                "set it in the environment or .env file"
            )
        return self.openai_api_key

    def check_qdrant(
        self,
        client_factory: Callable[..., QdrantHealthClient] | None = None,
    ) -> None:
        """Verify Qdrant connectivity and normalize transport failures."""

        if client_factory is None:
            from qdrant_client import QdrantClient

            client_factory = QdrantClient

        try:
            client = client_factory(url=self.qdrant_url, timeout=5)
            client.get_collections()
        except Exception as error:
            raise QdrantUnavailableError(
                f"Qdrant is unavailable at {self.qdrant_url}: {error}"
            ) from error
