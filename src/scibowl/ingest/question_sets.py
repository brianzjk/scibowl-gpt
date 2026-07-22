from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from scibowl.schema.common import AnswerMode, Category, QuestionType, SourceType
from scibowl.schema.question import AnswerGuidance, Choice, NormalizedQuestion, Provenance, SourceMetadata
from scibowl.utils.ids import slugify
from scibowl.utils.io import read_jsonl
from scibowl.utils.text import normalize_whitespace


def load_normalized_questions(path: Path) -> list[NormalizedQuestion]:
    return read_jsonl(path, NormalizedQuestion)


def load_question_specs(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalize_category(value: str) -> Category:
    normalized = slugify(value)
    category_map = {
        "biology": Category.BIOLOGY,
        "chemistry": Category.CHEMISTRY,
        "physics": Category.PHYSICS,
        "astronomy": Category.EARTH_SPACE,
        "earth": Category.EARTH_SPACE,
        "earth_science": Category.EARTH_SPACE,
        "earth_and_space": Category.EARTH_SPACE,
        "earth_space": Category.EARTH_SPACE,
        "ess": Category.EARTH_SPACE,
        "earth_and_space_science": Category.EARTH_SPACE,
        "space": Category.EARTH_SPACE,
        "math": Category.MATH,
        "energy": Category.ENERGY,
        "machine_learning": Category.ENERGY,
        "statistics": Category.ENERGY,
        "theoretical_cs": Category.ENERGY,
    }
    if normalized not in category_map:
        raise ValueError(f"Unsupported category: {value}")
    return category_map[normalized]


def _normalize_question_type(value: str) -> QuestionType:
    normalized = slugify(value)
    return QuestionType.BONUS if normalized == "bonus" else QuestionType.TOSSUP


def _is_visual_bonus(value: str) -> bool:
    return "visual bonus" in normalize_whitespace(value).lower()


def _normalize_answer_mode(value: str) -> AnswerMode:
    normalized = slugify(value)
    if normalized in {"multiple_choice", "multiplechoice"}:
        return AnswerMode.MULTIPLE_CHOICE
    if normalized in {"short_answer", "shortanswer"}:
        return AnswerMode.SHORT_ANSWER
    return AnswerMode.SHORT_ANSWER


def _difficulty_from_division(value: str) -> int:
    normalized = slugify(value)
    if normalized in {"rr", "round_robin"}:
        return 2
    if normalized in {"de", "double_elimination"}:
        return 4
    if normalized in {"wildcard", "playoff"}:
        return 5
    return 3


def _split_guidance(value: str) -> list[str]:
    normalized = normalize_whitespace(str(value))
    if not normalized:
        return []
    parts = [part.strip() for part in normalized.split(";")]
    return [part for part in parts if part]


def normalize_mit_question_csv(input_path: Path, source_id: str = "mit_2025") -> list[NormalizedQuestion]:
    df = pd.read_csv(input_path)
    rows: list[NormalizedQuestion] = []

    for index, row in df.fillna("").iterrows():
        if _is_visual_bonus(str(row.get("Type", ""))):
            continue
        category = _normalize_category(str(row["Category"]))
        question_type = _normalize_question_type(str(row["Type"]))
        answer_mode = _normalize_answer_mode(str(row["Format"]))
        difficulty = _difficulty_from_division(str(row.get("Division (Approx)", "")))

        question_text = normalize_whitespace(str(row["Question"]))
        answer_raw = normalize_whitespace(str(row["Answer"]))
        choices: list[Choice] = []

        if answer_mode == AnswerMode.MULTIPLE_CHOICE:
            for label in ["W", "X", "Y", "Z"]:
                choice_text = normalize_whitespace(str(row.get(label, "")))
                if choice_text:
                    choices.append(Choice(label=label, text=choice_text))
            if len(answer_raw) == 1:
                matched = next((choice for choice in choices if choice.label == answer_raw), None)
                answer_text = f"ANSWER: {matched.label}) {matched.text}" if matched else f"ANSWER: {answer_raw}"
            else:
                answer_text = f"ANSWER: {answer_raw}"
        else:
            answer_text = f"ANSWER: {answer_raw}"

        rows.append(
            NormalizedQuestion(
                question_id=f"{source_id}_{index + 1:04d}",
                source_type=SourceType.DATASET,
                source_id=source_id,
                category=category,
                subcategory=normalize_whitespace(str(row.get("Subcategory", "other"))) or "other",
                question_type=question_type,
                answer_mode=answer_mode,
                difficulty=difficulty,
                question_text=question_text,
                answer_text=answer_text,
                choices=choices,
                answer_guidance=AnswerGuidance(
                    accept=_split_guidance(row.get("Accept", "")),
                    do_not_accept=_split_guidance(row.get("Do Not Accept", "")),
                ),
                style_tags=["science_bowl", "mit"],
                content_tags=[tag for tag in [normalize_whitespace(str(row.get("Subcategory", "")))] if tag],
                source_metadata=SourceMetadata(
                    source_row=index + 1,
                    round=int(row["Round"]) if str(row.get("Round", "")).isdigit() else None,
                    tournament="MIT Science Bowl 2025",
                    year=2025,
                    original_difficulty=None,
                    original_quality=None,
                    writer=normalize_whitespace(str(row.get("Writer", ""))) or None,
                    source=normalize_whitespace(str(row.get("Source", ""))) or None,
                    date=normalize_whitespace(str(row.get("Date", ""))) or None,
                    division=normalize_whitespace(str(row.get("Division (Approx)", ""))) or None,
                ),
                provenance=Provenance(
                    raw_file=str(input_path),
                    parser_version="mit_csv_v2",
                    review_status="unreviewed",
                ),
            )
        )
    return rows


def normalize_mit_writing_directory(input_dir: Path, source_id: str = "mit_2025") -> list[NormalizedQuestion]:
    rows: list[NormalizedQuestion] = []
    question_index = 1

    for csv_path in sorted(input_dir.glob("*.csv")):
        for row_index, row in enumerate(_read_csv_rows(csv_path), start=1):
            if _is_visual_bonus(str(row.get("Type", ""))):
                continue
            question_text = normalize_whitespace(row.get("Question", ""))
            category_text = normalize_whitespace(row.get("Category", ""))
            if not question_text or not category_text:
                continue

            answer_mode = _normalize_answer_mode(str(row.get("Format", "")))
            answer_raw = normalize_whitespace(str(row.get("Answer", "")))
            if not answer_raw:
                continue

            choices: list[Choice] = []
            if answer_mode == AnswerMode.MULTIPLE_CHOICE:
                for label in ["W", "X", "Y", "Z"]:
                    choice_text = normalize_whitespace(str(row.get(label, "")))
                    if choice_text:
                        choices.append(Choice(label=label, text=choice_text))
                if len(answer_raw) == 1:
                    matched = next((choice for choice in choices if choice.label == answer_raw), None)
                    answer_text = f"ANSWER: {matched.label}) {matched.text}" if matched else f"ANSWER: {answer_raw}"
                else:
                    answer_text = f"ANSWER: {answer_raw}"
            else:
                answer_text = f"ANSWER: {answer_raw}"

            original_difficulty = _coerce_float(str(row.get("Difficulty", "")))
            rounded_difficulty = round(original_difficulty) if original_difficulty is not None and 1 <= original_difficulty <= 7 else 3
            original_quality = _coerce_float(str(row.get("Quality", "")))

            normalized_question = NormalizedQuestion(
                question_id=f"{source_id}_{question_index:04d}",
                source_type=SourceType.DATASET,
                source_id=source_id,
                category=_normalize_category(category_text),
                subcategory=normalize_whitespace(str(row.get("Subcategory", row.get("Topic", "other")))) or "other",
                question_type=_normalize_question_type(str(row.get("Type", ""))),
                answer_mode=answer_mode,
                difficulty=int(rounded_difficulty),
                question_text=question_text,
                answer_text=answer_text,
                choices=choices,
                answer_guidance=AnswerGuidance(
                    accept=_split_guidance(row.get("Accept", row.get("Also accept", ""))),
                    do_not_accept=_split_guidance(row.get("Do Not Accept", row.get("Do not accept", ""))),
                ),
                style_tags=["science_bowl", "mit", "writing_sheet"],
                content_tags=[
                    tag
                    for tag in [
                        normalize_whitespace(str(row.get("Subcategory", ""))),
                        normalize_whitespace(str(row.get("Topic", ""))),
                    ]
                    if tag
                ],
                source_metadata=SourceMetadata(
                    source_row=row_index,
                    round=int(row["Round"]) if str(row.get("Round", "")).isdigit() else None,
                    tournament="MIT Science Bowl 2025",
                    year=2025,
                    original_difficulty=original_difficulty,
                    original_quality=original_quality,
                    writer=normalize_whitespace(str(row.get("Writer", ""))) or None,
                    source=normalize_whitespace(str(row.get("Source", row.get("Background", "")))) or None,
                    date=normalize_whitespace(str(row.get("Date", ""))) or None,
                    division=normalize_whitespace(str(row.get("Division (Approx)", ""))) or None,
                ),
                provenance=Provenance(
                    raw_file=str(csv_path),
                    parser_version="mit_writing_dir_v1",
                    review_status="unreviewed",
                ),
            )
            rows.append(normalized_question)
            question_index += 1

    return rows


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        raw_header = next(reader)
        header = _dedupe_header(raw_header)
        return [dict(zip(header, row)) for row in reader]


def _dedupe_header(header: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    output: list[str] = []
    for index, name in enumerate(header):
        normalized = name.strip() or f"unnamed_{index}"
        count = counts.get(normalized, 0)
        output.append(normalized if count == 0 else f"{normalized}_{count + 1}")
        counts[normalized] = count + 1
    return output


def _coerce_float(value: str) -> float | None:
    text = normalize_whitespace(value)
    if not text or text in {"#DIV/0!", "?"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None
