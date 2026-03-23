from __future__ import annotations

from pathlib import Path

from scibowl.eval.benchmark import build_evaluation_record
from scibowl.generate.orchestration import GenerationOrchestrator
from scibowl.schema.dataset import EvaluationRecord
from scibowl.schema.generation import QuestionSpec
from scibowl.schema.question import NormalizedQuestion
from scibowl.schema.review import HumanReview
from scibowl.schema.textbook import TextbookChunk
from scibowl.utils.ids import make_id
from scibowl.utils.io import read_json, read_jsonl


def run_baseline_eval(
    *,
    split_path: Path,
    split_name: str,
    output_dir: Path,
    questions_path: Path | None = None,
    reviews_path: Path | None = None,
    textbook_chunks_path: Path | None = None,
    max_items: int | None = None,
) -> list[EvaluationRecord]:
    split = read_json(split_path)
    questions = read_jsonl(Path(questions_path or split["question_set_path"]), NormalizedQuestion)
    reviews = read_jsonl(Path(reviews_path or split["reviews_path"]), HumanReview)
    textbook_chunks = read_jsonl(textbook_chunks_path, TextbookChunk) if textbook_chunks_path else []

    split_ids = split[f"{split_name}_question_ids"]
    if max_items is not None:
        split_ids = split_ids[:max_items]

    question_lookup = {question.question_id: question for question in questions}
    held_out_questions = [question_lookup[question_id] for question_id in split_ids if question_id in question_lookup]
    held_out_id_set = {question.question_id for question in held_out_questions}
    style_questions = [question for question in questions if question.question_id not in held_out_id_set]
    review_lookup = _aggregate_reviews(reviews)

    orchestrator = GenerationOrchestrator()
    records: list[EvaluationRecord] = []
    for question in held_out_questions:
        spec = QuestionSpec(
            spec_id=make_id("spec"),
            category=question.category,
            subcategory=question.subcategory,
            question_type=question.question_type,
            answer_mode=question.answer_mode,
            difficulty=question.difficulty,
            topic_focus=question.content_tags or [question.subcategory],
            style_target_ids=[],
        )
        _, draft, report = orchestrator.run(spec, textbook_chunks, style_questions)
        record = build_evaluation_record("baseline_eval", question.question_id, spec, draft, report)
        aggregate = review_lookup.get(question.question_id)
        if aggregate:
            record.human_review = {
                "reviewed": True,
                "quality_mean": aggregate["quality_mean"],
                "difficulty_mean": aggregate["difficulty_mean"],
                "num_reviews": aggregate["count"],
            }
        records.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    from scibowl.utils.io import write_jsonl

    write_jsonl(output_dir / f"{split_name}_baseline_eval.jsonl", records)
    return records


def _aggregate_reviews(reviews: list[HumanReview]) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for review in reviews:
        bucket = grouped.setdefault(review.question_id, {"quality": [], "difficulty": []})
        if review.ratings.quality is not None:
            bucket["quality"].append(review.ratings.quality)
        if review.ratings.difficulty is not None:
            bucket["difficulty"].append(review.ratings.difficulty)

    output: dict[str, dict[str, float | int | None]] = {}
    for question_id, bucket in grouped.items():
        quality = bucket["quality"]
        difficulty = bucket["difficulty"]
        output[question_id] = {
            "quality_mean": sum(quality) / len(quality) if quality else None,
            "difficulty_mean": sum(difficulty) / len(difficulty) if difficulty else None,
            "count": max(len(quality), len(difficulty)),
        }
    return output
