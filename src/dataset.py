"""Streaming JSONL validation and the transcript-to-vector contract."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import parse_qs, urlparse


POINT_ID_NAMESPACE = uuid.UUID("8b10f755-39fb-4dc7-ab51-30bdbca44355")
SOURCE_FIELDS = (
    "chunk_id",
    "video_id",
    "chunk_index",
    "text",
    "start_time",
    "end_time",
    "formatted_time",
    "timestamp_link",
    "video_title",
    "channel",
    "video_url",
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    line_number: int
    reason: str

    def __str__(self) -> str:
        return f"line {self.line_number}: {self.reason}"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    total_lines: int
    valid_records: int
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues


class DatasetValidationError(ValueError):
    """Raised when one or more JSONL lines violate the indexing contract."""

    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
        self.issues = issues
        details = "; ".join(str(issue) for issue in issues)
        super().__init__(f"dataset validation failed: {details}")


@dataclass(frozen=True, slots=True)
class TranscriptRecord:
    """A validated record that keeps the original JSON payload intact."""

    data: Mapping[str, Any]

    @property
    def chunk_id(self) -> str:
        return str(self.data["chunk_id"])

    @property
    def video_id(self) -> str:
        return str(self.data["video_id"])

    @property
    def chunk_index(self) -> int:
        return int(self.data["chunk_index"])

    @property
    def text(self) -> str:
        return str(self.data["text"])

    @property
    def timestamp_link(self) -> str:
        return str(self.data["timestamp_link"])

    @property
    def embedding_input(self) -> str:
        title = str(self.data["video_title"]).strip()
        return f"{title}\n\n{self.text}"

    def qdrant_payload(self, *, embedding_model: str, dimensions: int) -> dict[str, Any]:
        """Return all source data plus explicit index compatibility metadata."""

        payload = dict(self.data)
        payload["index_metadata"] = {
            "embedding_model": embedding_model,
            "vector_dimensions": dimensions,
        }
        return payload


def point_id_for_chunk(chunk_id: str) -> str:
    """Map a stable chunk identifier to a deterministic Qdrant UUID."""

    normalized = chunk_id.strip()
    if not normalized:
        raise ValueError("chunk_id must not be empty")
    return str(uuid.uuid5(POINT_ID_NAMESPACE, normalized))


def _required_string(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def validate_record(data: object) -> TranscriptRecord:
    """Validate one decoded JSON object without importing any service SDK."""

    if not isinstance(data, dict):
        raise ValueError("record must be a JSON object")

    for field in (
        "chunk_id",
        "video_id",
        "text",
        "formatted_time",
        "timestamp_link",
        "video_title",
        "channel",
        "video_url",
    ):
        _required_string(data, field)

    chunk_index = data.get("chunk_index")
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise ValueError("chunk_index must be a non-negative integer")

    for field in ("start_time", "end_time"):
        value = data.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{field} must be a non-negative number")
    if data["end_time"] < data["start_time"]:
        raise ValueError("end_time must not precede start_time")

    timestamp = urlparse(str(data["timestamp_link"]))
    if timestamp.scheme not in {"http", "https"} or not timestamp.netloc:
        raise ValueError("timestamp_link must be an absolute HTTP(S) URL")
    if "t" not in parse_qs(timestamp.query):
        raise ValueError("timestamp_link must include an exact t query parameter")

    video_url = urlparse(str(data["video_url"]))
    if video_url.scheme not in {"http", "https"} or not video_url.netloc:
        raise ValueError("video_url must be an absolute HTTP(S) URL")

    return TranscriptRecord(data=dict(data))


def _parse_line(raw_line: str, line_number: int) -> TranscriptRecord:
    if not raw_line.strip():
        raise DatasetValidationError(
            (ValidationIssue(line_number, "blank lines are not valid JSON records"),)
        )
    try:
        data = json.loads(raw_line)
    except json.JSONDecodeError as error:
        raise DatasetValidationError(
            (ValidationIssue(line_number, f"malformed JSON: {error.msg}"),)
        ) from error
    try:
        return validate_record(data)
    except ValueError as error:
        raise DatasetValidationError((ValidationIssue(line_number, str(error)),)) from error


def iter_jsonl(path: str | Path) -> Iterator[TranscriptRecord]:
    """Yield validated records incrementally, stopping at the first invalid line."""

    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            yield _parse_line(raw_line, line_number)


def scan_jsonl(path: str | Path) -> ValidationReport:
    """Validate a complete file while retaining only IDs and error summaries."""

    issues: list[ValidationIssue] = []
    seen_ids: dict[str, int] = {}
    valid_records = 0
    total_lines = 0

    with Path(path).open("r", encoding="utf-8") as source:
        for total_lines, raw_line in enumerate(source, start=1):
            try:
                record = _parse_line(raw_line, total_lines)
            except DatasetValidationError as error:
                issues.extend(error.issues)
                continue

            previous_line = seen_ids.get(record.chunk_id)
            if previous_line is not None:
                issues.append(
                    ValidationIssue(
                        total_lines,
                        f"duplicate chunk_id {record.chunk_id!r}; first seen on line {previous_line}",
                    )
                )
                continue
            seen_ids[record.chunk_id] = total_lines
            valid_records += 1

    if total_lines == 0:
        issues.append(ValidationIssue(0, "dataset contains no JSONL records"))

    return ValidationReport(total_lines, valid_records, tuple(issues))


def require_valid_jsonl(path: str | Path) -> ValidationReport:
    """Preflight an entire dataset so invalid input cannot be partly indexed."""

    report = scan_jsonl(path)
    if report.issues:
        raise DatasetValidationError(report.issues)
    return report
