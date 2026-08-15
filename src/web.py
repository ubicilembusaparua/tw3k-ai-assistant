"""Local FastAPI application for grounded TW3K questions."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

from src.config import MissingConfigurationError, QdrantUnavailableError, Settings
from src.generation import GenerationServiceError, GroundedAnswer
from src.service import AssistantService, DuplicateSubmissionError, EmptyIndexError


ROOT = Path(__file__).parents[1]
templates = Jinja2Templates(directory=ROOT / "templates")


class Service(Protocol):
    def readiness(self) -> dict[str, object]: ...

    def ask(self, question: str) -> GroundedAnswer: ...


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


def _answer_payload(answer: GroundedAnswer) -> dict[str, object]:
    return {
        "status": answer.status,
        "text": answer.text,
        "claims": [
            {
                "text": claim.text,
                "citation_ids": [citation.identifier for citation in claim.citations],
            }
            for claim in answer.claims
        ],
        "sources": [
            {
                "id": source.identifier,
                "timestamp_link": source.timestamp_link,
                "video_title": source.video_title,
                "formatted_time": source.formatted_time,
                "excerpt": source.excerpt,
            }
            for source in answer.sources
        ],
    }


def _service_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


def create_app(
    settings: Settings | None = None,
    *,
    service: Service | None = None,
) -> FastAPI:
    app = FastAPI(title="TW3K AI Assistant")
    assistant = service or AssistantService(settings or Settings.from_env())
    app.state.assistant = assistant

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name="index.html")

    @app.get("/api/readiness")
    def readiness() -> dict[str, object]:
        return assistant.readiness()

    @app.post("/api/ask")
    def ask(payload: QuestionRequest) -> dict[str, object]:
        try:
            return _answer_payload(assistant.ask(payload.question))
        except DuplicateSubmissionError as error:
            raise _service_error(409, "duplicate_submission", str(error)) from error
        except EmptyIndexError as error:
            raise _service_error(409, "empty_index", str(error)) from error
        except QdrantUnavailableError as error:
            raise _service_error(503, "qdrant_error", str(error)) from error
        except MissingConfigurationError as error:
            raise _service_error(503, "openai_error", str(error)) from error
        except GenerationServiceError as error:
            raise _service_error(502, "openai_error", str(error)) from error

    return app


app = create_app()
