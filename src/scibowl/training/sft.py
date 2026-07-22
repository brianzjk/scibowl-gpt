from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from scibowl.prompts.category_guidance import category_prompt_guidance
from scibowl.schema.common import AnswerMode, Category, QuestionType, SourceType
from scibowl.schema.generation import QuestionSpec
from scibowl.schema.question import Choice, NormalizedQuestion
from scibowl.schema.training import ChatMessage, SFTExample, SFTMetadata
from scibowl.utils.ids import slugify
from scibowl.utils.io import read_jsonl, write_json, write_jsonl
from scibowl.utils.subcategories import canonicalize_subcategory
from scibowl.utils.subcategories import subcategory_guidance_terms
from scibowl.utils.text import normalize_whitespace


@dataclass(frozen=True)
class _PreparedQuestion:
    question: NormalizedQuestion
    sample_weight: float
    weight_reasons: tuple[str, ...]
    round_number: int | None
    fingerprint: str


def export_sft_dataset(
    questions_path: Path,
    output_dir: Path,
    validation_fraction: float = 0.05,
    seed: int = 42,
    excluded_categories: tuple[str, ...] = ("energy",),
) -> dict[str, object]:
    questions = read_jsonl(questions_path, NormalizedQuestion)
    excluded_category_set = {value.strip().lower() for value in excluded_categories}

    filtered: list[_PreparedQuestion] = []
    excluded_counts: dict[str, int] = {
        "excluded_category": 0,
        "missing_text": 0,
        "invalid_answer_line": 0,
        "contaminated_parse": 0,
        "malformed_multiple_choice": 0,
        "malformed_short_answer": 0,
    }

    for question in questions:
        exclusion_reason = _excluded_reason(question, excluded_category_set)
        if exclusion_reason is not None:
            excluded_counts[exclusion_reason] += 1
            continue

        sample_weight, weight_reasons = _sample_weight(question)
        filtered.append(
            _PreparedQuestion(
                question=question,
                sample_weight=sample_weight,
                weight_reasons=tuple(weight_reasons),
                round_number=_extract_round_number(question),
                fingerprint=_fingerprint_question(question),
            )
        )

    unique_questions, dedupe_count = _dedupe_questions(filtered)

    canonical_examples: list[SFTExample] = []
    train_examples: list[SFTExample] = []
    val_examples: list[SFTExample] = []
    materialized_train_examples: list[SFTExample] = []

    category_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {"train": 0, "val": 0}
    materialized_train_count = 0

    for prepared in unique_questions:
        split = _assign_split(prepared.question.question_id, seed=seed, validation_fraction=validation_fraction)
        repeat_count = _materialized_repeat_count(prepared.question.question_id, prepared.sample_weight, seed=seed)
        example = _build_sft_example(prepared, split=split, repeat_count=repeat_count)
        canonical_examples.append(example)
        split_counts[split] += 1
        category_counts[prepared.question.category.value] = category_counts.get(prepared.question.category.value, 0) + 1

        if split == "train":
            train_examples.append(example)
            materialized_train_examples.extend(_materialize_example(example))
            materialized_train_count += repeat_count
        else:
            val_examples.append(example)

    write_jsonl(output_dir / "sft_examples_all.jsonl", canonical_examples)
    write_jsonl(output_dir / "sft_examples_train.jsonl", train_examples)
    write_jsonl(output_dir / "sft_examples_val.jsonl", val_examples)
    write_jsonl(output_dir / "sft_examples_train_materialized.jsonl", materialized_train_examples)

    summary = {
        "input_questions": len(questions),
        "filtered_questions": len(filtered),
        "unique_questions": len(unique_questions),
        "deduplicated_questions": dedupe_count,
        "excluded_counts": excluded_counts,
        "split_counts": split_counts,
        "materialized_train_count": materialized_train_count,
        "category_counts": category_counts,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "excluded_categories": sorted(excluded_category_set),
        "weight_rules": {
            "default": 1.0,
            "mit_any_year": 2.0,
            "official_nsb_round_gt_11": 1.5,
        },
    }
    write_json(output_dir / "sft_summary.json", summary)
    return summary


def _excluded_reason(question: NormalizedQuestion, excluded_categories: set[str]) -> str | None:
    if question.category.value in excluded_categories:
        return "excluded_category"
    if not normalize_whitespace(question.question_text) or not normalize_whitespace(question.answer_text):
        return "missing_text"
    if not normalize_whitespace(question.answer_text).startswith("ANSWER:"):
        return "invalid_answer_line"
    if _looks_contaminated(question):
        return "contaminated_parse"

    if question.answer_mode == AnswerMode.MULTIPLE_CHOICE:
        if len(question.choices) != 4:
            return "malformed_multiple_choice"
        labels = [choice.label for choice in question.choices]
        if labels != ["W", "X", "Y", "Z"]:
            return "malformed_multiple_choice"
        if "Which of the following" not in question.question_text:
            return "malformed_multiple_choice"
        return None

    if question.choices:
        return "malformed_short_answer"
    return None


def _looks_contaminated(question: NormalizedQuestion) -> bool:
    question_text = normalize_whitespace(question.question_text)
    answer_text = normalize_whitespace(question.answer_text)
    answer_body = answer_text.removeprefix("ANSWER:").strip()

    if "ANSWER:" in question_text:
        return True
    if "ANSWER:" in answer_body:
        return True

    contamination_markers = (
        "toss-up",
        "tossup",
        "bonus",
        "multiple choice",
        "short answer",
        "science bowl",
    )
    lowered_answer = answer_body.casefold()
    return any(marker in lowered_answer for marker in contamination_markers)


def _sample_weight(question: NormalizedQuestion) -> tuple[float, list[str]]:
    weight = 1.0
    reasons: list[str] = []

    if question.source_id.startswith("mit_"):
        weight += 1.0
        reasons.append("mit")

    if question.source_id.startswith("nsb_set_") and (_extract_round_number(question) or 0) > 11:
        weight += 0.5
        reasons.append("late_official_nsb_round")

    return weight, reasons


def _extract_round_number(question: NormalizedQuestion) -> int | None:
    if question.source_metadata.round is not None:
        return question.source_metadata.round

    for candidate in (question.question_id, question.provenance.raw_file or ""):
        match = re.search(r"(?:round|rr|de)_(\d{1,2})", candidate, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _fingerprint_question(question: NormalizedQuestion) -> str:
    question_text = normalize_whitespace(question.question_text).casefold()
    answer_text = normalize_whitespace(question.answer_text).casefold()
    choices = "|".join(
        f"{choice.label.casefold()}:{normalize_whitespace(choice.text).casefold()}"
        for choice in question.choices
    )
    payload = f"{question_text}||{answer_text}||{choices}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _dedupe_priority(prepared: _PreparedQuestion) -> tuple[float, int, float, int, int]:
    question = prepared.question
    original_quality = question.source_metadata.original_quality
    has_quality = 1 if original_quality is not None else 0
    quality_value = original_quality if original_quality is not None else -999.0
    is_dataset = 1 if question.source_type == SourceType.DATASET else 0
    has_specific_subcategory = 1 if canonicalize_subcategory(question.subcategory).casefold() not in {"", "other"} else 0
    return (
        prepared.sample_weight,
        is_dataset,
        quality_value,
        has_quality,
        has_specific_subcategory,
    )


def _dedupe_questions(questions: list[_PreparedQuestion]) -> tuple[list[_PreparedQuestion], int]:
    best_by_fingerprint: dict[str, _PreparedQuestion] = {}

    for prepared in questions:
        incumbent = best_by_fingerprint.get(prepared.fingerprint)
        if incumbent is None:
            best_by_fingerprint[prepared.fingerprint] = prepared
            continue

        incumbent_priority = _dedupe_priority(incumbent)
        candidate_priority = _dedupe_priority(prepared)
        if candidate_priority > incumbent_priority:
            best_by_fingerprint[prepared.fingerprint] = prepared
        elif candidate_priority == incumbent_priority and prepared.question.question_id < incumbent.question.question_id:
            best_by_fingerprint[prepared.fingerprint] = prepared

    dedupe_count = len(questions) - len(best_by_fingerprint)
    unique_questions = sorted(best_by_fingerprint.values(), key=lambda item: item.question.question_id)
    return unique_questions, dedupe_count


def _assign_split(question_id: str, seed: int, validation_fraction: float) -> str:
    score = _stable_fraction(f"split:{seed}:{question_id}")
    return "val" if score < validation_fraction else "train"


def _materialized_repeat_count(question_id: str, sample_weight: float, seed: int) -> int:
    base_count = int(sample_weight)
    remainder = sample_weight - base_count
    if remainder <= 0:
        return max(1, base_count)
    threshold = _stable_fraction(f"repeat:{seed}:{question_id}")
    return base_count + (1 if threshold < remainder else 0)


def _stable_fraction(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    numerator = int.from_bytes(digest[:8], "big")
    return numerator / float(1 << 64)


def _build_sft_example(prepared: _PreparedQuestion, split: str, repeat_count: int) -> SFTExample:
    question = prepared.question
    canonical_subcategory = canonicalize_subcategory(question.subcategory) or question.subcategory
    topic_focus = question.content_tags or ([canonical_subcategory] if canonical_subcategory and canonical_subcategory.casefold() != "other" else [])
    spec = QuestionSpec(
        spec_id=f"sft_spec_{slugify(question.question_id)}",
        category=question.category,
        subcategory=canonical_subcategory or question.subcategory,
        question_type=question.question_type,
        difficulty=question.difficulty,
        topic_focus=topic_focus,
    )
    system_prompt = _render_sft_system_prompt()
    user_prompt = _render_sft_user_prompt(spec)
    assistant_content = json.dumps(
        {
            "question_text": question.question_text,
            "answer_text": question.answer_text,
            "choices": [choice.model_dump() for choice in question.choices],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return SFTExample(
        example_id=f"sft_{slugify(question.question_id)}",
        messages=[
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
            ChatMessage(role="assistant", content=assistant_content),
        ],
        metadata=SFTMetadata(
            source_question_id=question.question_id,
            source_dataset=question.source_id,
            source_tournament=question.source_metadata.tournament or question.source_id,
            source_round_number=prepared.round_number,
            category=question.category,
            subcategory=canonical_subcategory or question.subcategory,
            question_type=question.question_type,
            target_answer_mode=question.answer_mode,
            difficulty=question.difficulty,
            sample_weight=prepared.sample_weight,
            weight_reasons=list(prepared.weight_reasons),
            split=split,
            repeat_count=repeat_count if split == "train" else 1,
            has_specific_subcategory=(canonical_subcategory or question.subcategory).casefold() != "other",
            dedupe_fingerprint=prepared.fingerprint,
        ),
    )


def _materialize_example(example: SFTExample) -> list[SFTExample]:
    records: list[SFTExample] = []
    repeat_count = max(1, example.metadata.repeat_count)
    for index in range(repeat_count):
        materialized = example.model_copy(deep=True)
        materialized.example_id = f"{example.example_id}__rep{index + 1:02d}"
        records.append(materialized)
    return records


def _render_sft_system_prompt() -> str:
    return "\n".join(
        [
            "You are an MSB / Science Bowl question writer.",
            "Write one new question that matches Science Bowl conventions and the requested category/subcategory.",
            "Prefer conceptual questions with harder clues earlier and more obvious clues later.",
            "State the interrogative target early enough that players know what they are being asked for.",
            "Choose short-answer or multiple-choice based on what best fits the concept.",
            "For short-answer questions, use a concise answer line beginning with \"ANSWER:\" and an empty choices array.",
            "For multiple-choice questions, use exactly four choices labeled W, X, Y, Z, keep them out of question_text, and include the exact phrase \"Which of the following\" in question_text.",
            "Return only valid JSON with exactly these top-level keys: question_text, answer_text, choices.",
        ]
    )


def _render_sft_user_prompt(spec: QuestionSpec) -> str:
    category_guidance = category_prompt_guidance(spec.category)
    subcategory_guidance = ", ".join(subcategory_guidance_terms(spec.subcategory)) or "none"
    topic_focus = ", ".join(spec.topic_focus) or "none"
    guidance_lines = "\n".join(f"- {line}" for line in category_guidance[:3]) or "- none"
    return "\n".join(
        [
            f"Category: {spec.category.value}",
            "Category Guidance:",
            guidance_lines,
            f"Subcategory: {spec.subcategory}",
            f"Subcategory Guidance: {subcategory_guidance}",
            f"Type: {spec.question_type.value}",
            f"Difficulty: {spec.difficulty}",
            f"Topic Focus: {topic_focus}",
            "Write one new Science Bowl question and return only JSON.",
        ]
    )
