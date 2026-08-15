from __future__ import annotations

import pytest

from src.config import (
    ConfigurationError,
    MissingConfigurationError,
    QdrantUnavailableError,
    Settings,
)


def test_settings_have_documented_local_defaults() -> None:
    settings = Settings.from_env({})

    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "tw3k_transcripts"
    assert settings.embedding_dimensions == 384
    assert settings.retrieval_candidate_count == 20
    assert settings.final_context_count == 8


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("QDRANT_URL", "localhost:6333", "absolute"),
        ("RETRIEVAL_CANDIDATE_COUNT", "zero", "integer"),
        ("FINAL_CONTEXT_COUNT", "0", "greater than zero"),
        ("RELEVANCE_THRESHOLD", "1.1", "between 0 and 1"),
        ("QDRANT_COLLECTION", "", "must not be empty"),
    ],
)
def test_invalid_configuration_has_clear_error(
    name: str, value: str, message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        Settings.from_env({name: value})


def test_generation_reports_missing_openai_configuration() -> None:
    settings = Settings.from_env({})

    with pytest.raises(MissingConfigurationError, match="OPENAI_API_KEY"):
        settings.require_openai_api_key()


def test_qdrant_failure_is_distinct_from_missing_configuration() -> None:
    settings = Settings.from_env({})

    class UnavailableClient:
        def __init__(self, **_: object) -> None:
            pass

        def get_collections(self) -> object:
            raise ConnectionError("connection refused")

    with pytest.raises(QdrantUnavailableError, match="connection refused"):
        settings.check_qdrant(UnavailableClient)


def test_qdrant_health_check_uses_configured_endpoint() -> None:
    calls: list[tuple[str, int]] = []

    class HealthyClient:
        def __init__(self, *, url: str, timeout: int) -> None:
            calls.append((url, timeout))

        def get_collections(self) -> object:
            return object()

    Settings.from_env({"QDRANT_URL": "http://qdrant:6333"}).check_qdrant(
        HealthyClient
    )

    assert calls == [("http://qdrant:6333", 5)]
