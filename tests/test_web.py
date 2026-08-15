from __future__ import annotations

from fastapi.testclient import TestClient

from src.config import QdrantUnavailableError
from src.generation import AnswerClaim, Citation, GenerationServiceError, GroundedAnswer
from src.service import DuplicateSubmissionError, EmptyIndexError
from src.web import create_app


def citation() -> Citation:
    return Citation(
        identifier="SOURCE_1",
        timestamp_link="https://youtube.com/watch?v=video&t=12s",
        video_title="<script>unsafe title</script>",
        formatted_time="00:12 - 00:20",
        excerpt="<img src=x onerror=alert(1)>",
    )


class FakeService:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.questions: list[str] = []

    def readiness(self) -> dict[str, object]:
        return {
            "ready": False,
            "components": {
                "app": {"ready": True},
                "qdrant": {"ready": True, "index_count": 3},
                "embedding_model": {"ready": False},
                "reranker_model": {"ready": False},
                "openai": {"ready": False},
            },
        }

    def ask(self, question: str) -> GroundedAnswer:
        self.questions.append(question)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome  # type: ignore[return-value]


def client(outcome: object) -> TestClient:
    return TestClient(create_app(service=FakeService(outcome)))


def answered() -> GroundedAnswer:
    source = citation()
    return GroundedAnswer(
        status="answered",
        text="Use farms. [SOURCE_1](https://youtube.com/watch?v=video&t=12s)",
        claims=(AnswerClaim("Use farms.", (source,)),),
        sources=(source,),
    )


def test_home_has_responsive_safe_duplicate_protected_interface() -> None:
    response = client(answered()).get("/")

    assert response.status_code == 200
    assert 'name="viewport"' in response.text
    assert "submitting = false" in response.text
    assert "if (submitting) return" in response.text
    assert "submit.disabled = true" in response.text
    assert "textContent" in response.text
    assert "innerHTML" not in response.text
    assert "document.createElement('details')" in response.text


def test_readiness_reports_each_required_component() -> None:
    response = client(answered()).get("/api/readiness")

    assert response.status_code == 200
    components = response.json()["components"]
    assert set(components) == {
        "app",
        "qdrant",
        "embedding_model",
        "reranker_model",
        "openai",
    }


def test_ask_returns_structured_exact_timestamp_sources_without_rendering_html() -> None:
    response = client(answered()).post("/api/ask", json={"question": "food"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["claims"][0]["citation_ids"] == ["SOURCE_1"]
    assert payload["sources"][0]["timestamp_link"].endswith("&t=12s")
    assert payload["sources"][0]["excerpt"].startswith("<img")


def test_insufficient_evidence_is_not_a_service_failure() -> None:
    answer = GroundedAnswer(
        "insufficient_evidence",
        "Relevant information was not found in the available sources.",
        (),
        (),
    )

    response = client(answer).post("/api/ask", json={"question": "Neptune"})

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"


def test_blank_question_is_rejected_before_service_work() -> None:
    response = client(answered()).post("/api/ask", json={"question": "   "})

    assert response.status_code == 422


def test_service_failures_have_distinct_status_codes() -> None:
    cases = [
        (EmptyIndexError("empty"), 409, "empty_index"),
        (DuplicateSubmissionError("duplicate"), 409, "duplicate_submission"),
        (QdrantUnavailableError("offline"), 503, "qdrant_error"),
        (GenerationServiceError("provider"), 502, "openai_error"),
    ]
    for error, status_code, code in cases:
        response = client(error).post("/api/ask", json={"question": "food"})
        assert response.status_code == status_code
        assert response.json()["detail"]["code"] == code
