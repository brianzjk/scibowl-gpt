from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from scibowl.ingest.question_sets import _is_visual_bonus
from scibowl.schema.question import NormalizedQuestion
from scibowl.schema.review import HumanReview, ReviewRatings
from scibowl.utils.ids import make_id, slugify
from scibowl.utils.io import read_jsonl
from scibowl.utils.text import normalize_whitespace


QUESTION_ID_COLUMNS = ("question_id", "Question ID", "questionId")
SOURCE_ROW_COLUMNS = ("source_row", "Source Row", "row_number", "Row Number", "row")
REVIEWER_COLUMNS = ("reviewer_id", "Reviewer", "reviewer", "writer", "Writer")
QUALITY_COLUMNS = ("quality", "Quality", "quality_rating", "Quality Rating")
DIFFICULTY_COLUMNS = ("difficulty", "Difficulty", "difficulty_rating", "Difficulty Rating")
COMMENT_COLUMNS = ("comment", "Comment", "notes", "Notes")
STATUS_COLUMNS = ("status", "Status")
TOURNAMENT_COLUMNS = ("tournament", "Tournament")


def import_ratings_csv(
    input_path: Path,
    *,
    questions_path: Path | None = None,
    tournament: str | None = None,
) -> list[HumanReview]:
    df = pd.read_csv(input_path)
    question_lookup = _build_question_lookup(questions_path) if questions_path else {}

    reviews: list[HumanReview] = []
    for index, row in df.fillna("").iterrows():
        reviewer_id = normalize_whitespace(_pick_value(row, REVIEWER_COLUMNS))
        if not reviewer_id:
            reviewer_id = f"reviewer_{index + 1:04d}"

        source_row = _coerce_int(_pick_value(row, SOURCE_ROW_COLUMNS))
        question_id = normalize_whitespace(_pick_value(row, QUESTION_ID_COLUMNS))
        if not question_id and source_row is not None:
            question_id = question_lookup.get(source_row, "")
        if not question_id:
            raise ValueError(
                "Each ratings row must include question_id or source_row that can be linked via --questions"
            )

        reviews.append(
            HumanReview(
                review_id=make_id("review"),
                question_id=question_id,
                reviewer_id=slugify(reviewer_id) or reviewer_id,
                tournament=normalize_whitespace(_pick_value(row, TOURNAMENT_COLUMNS)) or tournament,
                ratings=ReviewRatings(
                    quality=_coerce_float(_pick_value(row, QUALITY_COLUMNS)),
                    difficulty=_coerce_float(_pick_value(row, DIFFICULTY_COLUMNS)),
                ),
                comment=normalize_whitespace(_pick_value(row, COMMENT_COLUMNS)) or None,
                status=normalize_whitespace(_pick_value(row, STATUS_COLUMNS)) or None,
                source_row=source_row,
                metadata={"raw_file": str(input_path)},
            )
        )
    return reviews


def import_mit_rating_directory(
    input_dir: Path,
    *,
    questions_path: Path,
    tournament: str = "MIT Science Bowl 2025",
) -> list[HumanReview]:
    questions = read_jsonl(questions_path, NormalizedQuestion)
    question_lookup = {
        normalize_whitespace(question.question_text).lower(): question
        for question in questions
    }

    reviews: list[HumanReview] = []
    for csv_path in sorted(input_dir.glob("*.csv")):
        for row_index, row in enumerate(_read_csv_rows(csv_path), start=1):
            if _is_visual_bonus(str(row.get("Type", ""))):
                continue
            question_text = normalize_whitespace(row.get("Question", ""))
            if not question_text:
                continue

            question = question_lookup.get(question_text.lower())
            if question is None:
                continue

            difficulty = _coerce_float(str(row.get("Difficulty", "")))
            quality = _coerce_float(str(row.get("Quality", "")))
            valid_difficulty = difficulty if difficulty is not None and 1 <= difficulty <= 7 else None
            valid_quality = quality if quality is not None and -1 <= quality <= 1 else None
            if valid_difficulty is None and valid_quality is None:
                continue

            reviews.append(
                _build_review(
                    question_id=question.question_id,
                    reviewer_id="aggregate_sheet_ratings",
                    tournament=tournament,
                    source_row=question.source_metadata.source_row,
                    quality=valid_quality,
                    difficulty=valid_difficulty,
                    source_file=csv_path,
                    rating_column="Difficulty,Quality",
                    row_index=row_index,
                )
            )
    return reviews


def _build_review(
    *,
    question_id: str,
    reviewer_id: str,
    tournament: str,
    source_row: int | None,
    quality: float | None,
    difficulty: float | None,
    source_file: Path,
    rating_column: str,
    row_index: int,
) -> HumanReview:
    return HumanReview(
        review_id=make_id("review"),
        question_id=question_id,
        reviewer_id=reviewer_id,
        tournament=tournament,
        ratings=ReviewRatings(quality=quality, difficulty=difficulty),
        source_row=source_row,
        metadata={
            "raw_file": str(source_file),
            "rating_column": rating_column,
            "sheet_row": row_index,
        },
    )


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




def _build_question_lookup(questions_path: Path) -> dict[int, str]:
    rows = read_jsonl(questions_path, NormalizedQuestion)
    lookup: dict[int, str] = {}
    for fallback_index, question in enumerate(rows, start=1):
        source_row = question.source_metadata.source_row or fallback_index
        lookup[source_row] = question.question_id
    return lookup


def _pick_value(row: pd.Series, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in row:
            return str(row[candidate])
    return ""


def _coerce_float(value: str) -> float | None:
    text = normalize_whitespace(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _coerce_int(value: str) -> int | None:
    text = normalize_whitespace(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None
