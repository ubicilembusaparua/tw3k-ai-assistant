from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.dataset import (
    DatasetValidationError,
    iter_jsonl,
    point_id_for_chunk,
    require_valid_jsonl,
    scan_jsonl,
    validate_record,
)


FIXTURE = Path(__file__).parent / "fixtures" / "transcripts.jsonl"


def valid_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "chunk_id": "abc_c0001",
        "video_id": "abc",
        "chunk_index": 1,
        "text": " Original transcript text. ",
        "start_time": 12.5,
        "end_time": 18.0,
        "formatted_time": "00:12 - 00:18",
        "timestamp_link": "https://youtube.com/watch?v=abc&t=12s",
        "video_title": "Campaign guide",
        "channel": "Guide channel",
        "video_url": "https://youtube.com/watch?v=abc",
        "custom_source_field": {"kept": True},
    }
    record.update(overrides)
    return record


def test_validation_has_no_service_sdk_import_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in ("qdrant_client", "sentence_transformers", "openai"):
        monkeypatch.setitem(sys.modules, module_name, None)

    report = require_valid_jsonl(FIXTURE)

    assert report.valid_records == 3
    assert report.total_lines == 3


def test_record_preserves_original_payload_and_builds_embedding_input() -> None:
    source = valid_record()
    record = validate_record(source)
    payload = record.qdrant_payload(embedding_model="test-model", dimensions=384)

    assert record.text == " Original transcript text. "
    assert record.embedding_input == "Campaign guide\n\n Original transcript text. "
    assert payload["custom_source_field"] == {"kept": True}
    assert payload["timestamp_link"] == source["timestamp_link"]
    assert payload["index_metadata"] == {
        "embedding_model": "test-model",
        "vector_dimensions": 384,
    }


def test_point_ids_are_stable_and_chunk_specific() -> None:
    assert point_id_for_chunk("abc_c0001") == point_id_for_chunk("abc_c0001")
    assert point_id_for_chunk("abc_c0001") != point_id_for_chunk("abc_c0002")


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"chunk_id": ""}, "chunk_id"),
        ({"video_id": None}, "video_id"),
        ({"chunk_index": -1}, "chunk_index"),
        ({"text": "   "}, "text"),
        ({"timestamp_link": "https://youtube.com/watch?v=abc"}, "t query"),
        ({"end_time": 1.0}, "end_time"),
    ],
)
def test_required_contract_fields_are_validated(
    overrides: dict[str, object], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        validate_record(valid_record(**overrides))


def test_scan_reports_every_bad_line_and_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    records = [
        json.dumps(valid_record()),
        "not-json",
        json.dumps(valid_record(chunk_id="second", text="")),
        json.dumps(valid_record()),
    ]
    path.write_text("\n".join(records), encoding="utf-8")

    report = scan_jsonl(path)

    assert report.valid_records == 1
    assert [issue.line_number for issue in report.issues] == [2, 3, 4]
    assert "malformed JSON" in report.issues[0].reason
    assert "duplicate chunk_id" in report.issues[2].reason


def test_preflight_rejects_invalid_file_before_iteration(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(valid_record(text="")), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="line 1.*text"):
        require_valid_jsonl(path)


def test_preflight_rejects_empty_dataset(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="no JSONL records"):
        require_valid_jsonl(path)


def test_fixture_is_yielded_incrementally() -> None:
    records = iter_jsonl(FIXTURE)

    assert next(records).chunk_id == "video-a_c0000"
    assert next(records).chunk_id == "video-a_c0001"
