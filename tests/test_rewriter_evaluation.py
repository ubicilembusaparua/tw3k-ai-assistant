import csv
from pathlib import Path

import pytest

from _evaluation.evaluate_rewriter import QUESTIONS, parse_judge_scores, save_results


def test_rewriter_benchmark_contains_fifteen_questions():
    assert len(QUESTIONS) == 15
    assert len({item.question_id for item in QUESTIONS}) == 15
    assert all(item.question and item.reference_answer for item in QUESTIONS)


def test_parse_judge_scores_accepts_json_fence():
    result = parse_judge_scores(
        '```json\n{"without_query_rewriter":"good", "with_query_rewriter":"bad"}\n```'
    )

    assert result == {
        "without_query_rewriter": "good",
        "with_query_rewriter": "bad",
    }


def test_parse_judge_scores_rejects_unknown_label():
    with pytest.raises(ValueError, match="invalid score"):
        parse_judge_scores(
            '{"without_query_rewriter":"excellent", "with_query_rewriter":"bad"}'
        )


def test_save_results_writes_only_binary_score_columns():
    output_path = Path.cwd() / ".rewriter-test-output.csv"
    try:
        output_path = save_results(
            [
                {
                    "question_id": "q01",
                    "question": "Example question",
                    "without_query_rewriter": "good",
                    "with_query_rewriter": "bad",
                }
            ],
            output_path,
        )

        with output_path.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))
    finally:
        output_path.unlink(missing_ok=True)

    assert rows == [
        {
            "question_id": "q01",
            "question": "Example question",
            "without_query_rewriter": "good",
            "with_query_rewriter": "bad",
        }
    ]
