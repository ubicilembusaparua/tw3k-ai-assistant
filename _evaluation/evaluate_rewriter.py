"""Evaluate the RAG pipeline with and without query rewriting.

The evaluator runs the same 15 questions through both application schemas and
uses an OpenAI model as a judge.  The output intentionally contains only the
binary labels requested for each schema; candidate answers are kept in memory
and are not written to the results CSV.

Run from the repository root with::

    uv run python _evaluation/evaluate_rewriter.py

The OpenAI API key is loaded from ``.env`` or the process environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv
from openai import OpenAI

# Allow the script to be run as ``python _evaluation/evaluate_rewriter.py``.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from query_rewriter import QueryRewriter
from rag_app import RAGBase


OUTPUT_PATH = ROOT_DIR / "_evaluation" / "results" / "rewriter.csv"
DEFAULT_MODEL = "gpt-5.4-mini"
ALLOWED_SCORES = {"good", "bad"}


@dataclass(frozen=True)
class EvaluationQuestion:
    """A benchmark question and the facts a good answer should contain."""

    question_id: str
    question: str
    reference_answer: str


# These cover a range of campaign, diplomacy, character, economy, and battle
# topics.  The references are evaluation criteria, not text that is supplied to
# either RAG schema when it generates its answer.
QUESTIONS: tuple[EvaluationQuestion, ...] = (
    EvaluationQuestion(
        "q01",
        "How do I keep my commanderies orderly while reducing corruption?",
        "State Workshops can reduce corruption in adjacent commanderies; lower tax rates and high-Authority administrators can improve public order.",
    ),
    EvaluationQuestion(
        "q02",
        "What does Cao Cao use to make other factions fight for him?",
        "Cao Cao uses Credibility to influence diplomatic relations and incite proxy wars between other factions without directly fighting those wars himself.",
    ),
    EvaluationQuestion(
        "q03",
        "How many generals and retinues can I have in one army?",
        "An army can contain up to three generals, and each general can command up to six retinues matched to the general's class and elemental color.",
    ),
    EvaluationQuestion(
        "q04",
        "How should I use cavalry to get behind enemy bows?",
        "Heavy shock cavalry can flank around spear infantry and charge vulnerable archers or crossbowmen from behind.",
    ),
    EvaluationQuestion(
        "q05",
        "What happens when my generals are unhappy?",
        "Low character satisfaction can lead to defections, spying for the enemy, or civil-war rebellions.",
    ),
    EvaluationQuestion(
        "q06",
        "How can I get more trade deals with rival warlords?",
        "Foreign trade reforms, market wharves, and a contiguous land border or shared port coastline can enable additional trade agreements.",
    ),
    EvaluationQuestion(
        "q07",
        "How does Dong Zhuo keep his Intimidation up?",
        "Dong Zhuo can maintain Intimidation by executing captured generals, razing settlements, and using harsh governance.",
    ),
    EvaluationQuestion(
        "q08",
        "What do spies need before they can perform actions in enemy factions?",
        "Spies build Cover and Network points over time; those points enable covert actions such as sabotage, assassination, or surrendering territory.",
    ),
    EvaluationQuestion(
        "q09",
        "What does assigning an Administrator do for a commandery?",
        "Administrators provide commandery-wide benefits to income, public order, and corruption, and help defend the city garrison during sieges.",
    ),
    EvaluationQuestion(
        "q10",
        "How do I gain Imperial Favor and avoid becoming an enemy of the Han?",
        "Defeating Han enemies, avoiding war with loyalist Han factions, and fulfilling Han Emperor mandates helps maintain Imperial Favor.",
    ),
    EvaluationQuestion(
        "q11",
        "What is the difference between shock cavalry and melee cavalry?",
        "Shock cavalry specialize in high charge impact against infantry, while melee cavalry have stronger missile-blocking capability and are useful for hunting ranged units.",
    ),
    EvaluationQuestion(
        "q12",
        "Which building chain gives the most industrial income?",
        "Private Artisans and State Workshops provide high base industrial income and improve industrial income multipliers.",
    ),
    EvaluationQuestion(
        "q13",
        "How can I produce more food for growing commanderies?",
        "Upgrading Land Development and livestock farms and researching agricultural reforms increases food production for trade and population growth.",
    ),
    EvaluationQuestion(
        "q14",
        "What is Yuan Shao's Lineage used for?",
        "Yuan Shao uses Lineage to recruit specialized captain retinues without needing a full general, making army recruitment cheaper.",
    ),
    EvaluationQuestion(
        "q15",
        "What should I do about Yellow Turban rebellions when public order is low?",
        "Garrison armies, public-order buildings such as grain stores or Confucian temples, and defeating Yellow Turban stacks before they capture county capitals help contain the rebellions.",
    ),
)


JUDGE_INSTRUCTIONS = """
You are a strict but fair evaluator of answers about Total War: Three Kingdoms.
Judge each candidate independently against the question and reference answer.

Score a candidate "good" only when it directly answers the question, includes
the central facts from the reference, and has no material factual errors or
unsupported claims. Minor wording differences or a small omission are allowed
when the main answer is correct. Score it "bad" when it is irrelevant, fails
to answer, contradicts the reference, hallucinates important details, or
refuses to answer a question that the reference covers.

The labels are independent: do not give one candidate a score merely because
it is better or worse than the other candidate. Return only valid JSON with
exactly these two keys and no markdown or explanation:
{"without_query_rewriter":"good|bad","with_query_rewriter":"good|bad"}
""".strip()


def build_assistant_pair(
    client: OpenAI,
    *,
    answer_model: str = DEFAULT_MODEL,
    rerank: bool = True,
) -> tuple[RAGBase, RAGBase]:
    """Build both schemas while sharing the retrieval and reranking objects."""

    without_rewriter = RAGBase(
        llm_client=client,
        model=answer_model,
        rerank=rerank,
        use_query_rewriter=False,
    )
    with_rewriter = RAGBase(
        index=without_rewriter.index,
        reranker=without_rewriter.reranker,
        llm_client=client,
        model=answer_model,
        query_rewriter=QueryRewriter(client, model=answer_model),
        use_query_rewriter=True,
    )
    return without_rewriter, with_rewriter


def answer_text(assistant: RAGBase, question: str) -> str:
    """Run one question and normalize the Responses API result to text."""

    response = assistant.rag(question)
    answer = getattr(response, "output_text", response)
    answer = str(answer).strip()
    if not answer:
        raise RuntimeError(f"The assistant returned an empty answer for: {question}")
    return answer


def build_judge_prompt(
    item: EvaluationQuestion,
    without_rewriter_answer: str,
    with_rewriter_answer: str,
) -> str:
    """Build the per-question prompt sent to the judge model."""

    return (
        f"QUESTION:\n{item.question}\n\n"
        f"REFERENCE ANSWER / FACT CHECK:\n{item.reference_answer}\n\n"
        "CANDIDATE A (without query rewriter):\n"
        f"{without_rewriter_answer}\n\n"
        "CANDIDATE B (with query rewriter):\n"
        f"{with_rewriter_answer}\n"
    )


def _score_value(value: Any, key: str) -> str:
    """Validate and normalize one judge score."""

    if isinstance(value, Mapping):
        value = value.get("score")
    score = str(value).strip().lower()
    if score not in ALLOWED_SCORES:
        raise ValueError(f"Judge returned an invalid score for {key!r}: {value!r}")
    return score


def parse_judge_scores(raw_output: str) -> dict[str, str]:
    """Parse the judge's JSON, tolerating a surrounding markdown fence."""

    raw_output = raw_output.strip()
    candidates = [raw_output]
    json_match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
    if json_match and json_match.group(0) != raw_output:
        candidates.append(json_match.group(0))

    parsed: Any = None
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_error = exc

    if not isinstance(parsed, Mapping):
        raise ValueError("Judge did not return a JSON object") from last_error

    required_keys = ("without_query_rewriter", "with_query_rewriter")
    missing = [key for key in required_keys if key not in parsed]
    if missing:
        raise ValueError(f"Judge response is missing score keys: {', '.join(missing)}")

    return {key: _score_value(parsed[key], key) for key in required_keys}


def judge_answers(
    client: OpenAI,
    item: EvaluationQuestion,
    without_rewriter_answer: str,
    with_rewriter_answer: str,
    *,
    judge_model: str = DEFAULT_MODEL,
) -> dict[str, str]:
    """Ask the OpenAI judge to label both candidate answers."""

    response = client.responses.create(
        model=judge_model,
        input=[
            {"role": "developer", "content": JUDGE_INSTRUCTIONS},
            {
                "role": "user",
                "content": build_judge_prompt(
                    item,
                    without_rewriter_answer,
                    with_rewriter_answer,
                ),
            },
        ],
    )
    return parse_judge_scores(response.output_text)


def evaluate_questions(
    client: OpenAI,
    questions: Sequence[EvaluationQuestion] = QUESTIONS,
    *,
    answer_model: str = DEFAULT_MODEL,
    judge_model: str = DEFAULT_MODEL,
    rerank: bool = True,
) -> list[dict[str, str]]:
    """Generate and judge answers for every benchmark question."""

    without_rewriter, with_rewriter = build_assistant_pair(
        client,
        answer_model=answer_model,
        rerank=rerank,
    )
    rows: list[dict[str, str]] = []

    for item in questions:
        print(f"Evaluating {item.question_id}: {item.question}")
        without_answer = answer_text(without_rewriter, item.question)
        with_answer = answer_text(with_rewriter, item.question)
        scores = judge_answers(
            client,
            item,
            without_answer,
            with_answer,
            judge_model=judge_model,
        )
        rows.append(
            {
                "question_id": item.question_id,
                "question": item.question,
                "without_query_rewriter": scores["without_query_rewriter"],
                "with_query_rewriter": scores["with_query_rewriter"],
            }
        )
        print(
            "  scores: "
            f"without_query_rewriter={scores['without_query_rewriter']}, "
            f"with_query_rewriter={scores['with_query_rewriter']}"
        )

    return rows


def save_results(rows: Sequence[Mapping[str, str]], output_path: Path = OUTPUT_PATH) -> Path:
    """Write the binary comparison scores to a CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "question",
        "without_query_rewriter",
        "with_query_rewriter",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fieldnames} for row in rows)
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="CSV path for the comparison results.",
    )
    parser.add_argument(
        "--answer-model",
        default=os.getenv("OPENAI_MODEL_NAME", DEFAULT_MODEL),
        help="Model used by both RAG answer schemas.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("OPENAI_JUDGE_MODEL", DEFAULT_MODEL),
        help="Model used to judge both answers.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable the existing cross-encoder reranker for both schemas.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    load_dotenv(ROOT_DIR / ".env")
    client = OpenAI()
    rows = evaluate_questions(
        client,
        answer_model=args.answer_model,
        judge_model=args.judge_model,
        rerank=not args.no_rerank,
    )
    output_path = save_results(rows, args.output)
    print(f"Saved {len(rows)} evaluation rows to {output_path.resolve()}")


if __name__ == "__main__":
    main()
