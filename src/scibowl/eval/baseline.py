from __future__ import annotations

from collections import Counter
from pathlib import Path

from scibowl.eval.benchmark import build_evaluation_record
from scibowl.generate.orchestration import GenerationOrchestrator
from scibowl.schema.dataset import BaselineRunRecord, EvaluationRecord
from scibowl.schema.generation import QuestionSpec
from scibowl.schema.question import NormalizedQuestion
from scibowl.schema.review import HumanReview
from scibowl.schema.textbook import TextbookChunk
from scibowl.utils.ids import make_id
from scibowl.utils.io import ensure_parent, read_json, read_jsonl, write_json


def run_baseline_eval(
    *,
    split_path: Path,
    split_name: str,
    output_dir: Path,
    questions_path: Path | None = None,
    style_questions_path: Path | None = None,
    reviews_path: Path | None = None,
    textbook_chunks_path: Path | None = None,
    max_items: int | None = None,
    max_per_category: int | None = None,
    excluded_categories: list[str] | None = None,
) -> list[EvaluationRecord]:
    split = read_json(split_path)
    questions = read_jsonl(Path(questions_path or split["question_set_path"]), NormalizedQuestion)
    style_questions = read_jsonl(Path(style_questions_path), NormalizedQuestion) if style_questions_path else questions
    reviews = read_jsonl(Path(reviews_path or split["reviews_path"]), HumanReview)
    textbook_chunks = _load_textbook_chunks(textbook_chunks_path) if textbook_chunks_path else []
    output_path = output_dir / f"{split_name}_baseline_eval.jsonl"
    runs_output_path = output_dir / f"{split_name}_baseline_runs.jsonl"
    existing_records = read_jsonl(output_path, EvaluationRecord) if output_path.exists() else []
    existing_run_records = read_jsonl(runs_output_path, BaselineRunRecord) if runs_output_path.exists() else []
    existing_eval_by_id = {record.candidate_question_id: record for record in existing_records}
    existing_eval_ids = {record.candidate_question_id for record in existing_records}
    existing_run_ids = {record.candidate_question_id for record in existing_run_records}

    split_ids = split[f"{split_name}_question_ids"]
    if max_items is not None:
        split_ids = split_ids[:max_items]

    question_lookup = {question.question_id: question for question in questions}
    held_out_questions = [question_lookup[question_id] for question_id in split_ids if question_id in question_lookup]
    excluded = {item.strip().lower() for item in (excluded_categories or ["energy"]) if item.strip()}
    held_out_questions = [question for question in held_out_questions if question.category.value not in excluded]
    if max_per_category is not None:
        held_out_questions = _limit_questions_per_category(held_out_questions, max_per_category)
    held_out_id_set = {question.question_id for question in held_out_questions}
    style_questions = [question for question in style_questions if question.question_id not in held_out_id_set]
    review_lookup = _aggregate_reviews(reviews)

    orchestrator = GenerationOrchestrator()
    records: list[EvaluationRecord] = list(existing_records)
    ensure_parent(output_path)
    ensure_parent(runs_output_path)
    if not output_path.exists():
        output_path.write_text("", encoding="utf-8")
    if not runs_output_path.exists():
        runs_output_path.write_text("", encoding="utf-8")
    for question in held_out_questions:
        has_eval = question.question_id in existing_eval_ids
        has_run = question.question_id in existing_run_ids
        if has_eval and has_run:
            continue
        spec = QuestionSpec(
            spec_id=make_id("spec"),
            category=question.category,
            subcategory=question.subcategory,
            question_type=question.question_type,
            answer_mode=None,
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
        record_for_run = existing_eval_by_id.get(question.question_id, record)
        if not has_eval:
            records.append(record)
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(record.model_dump_json() + "\n")
        if not has_run:
            run_record = BaselineRunRecord(
                run_id="baseline_eval",
                candidate_question_id=question.question_id,
                spec=spec,
                draft=draft,
                report=report,
                evaluation=record_for_run,
                reference_question=question,
                reference_human_review=record_for_run.human_review,
            )
            with runs_output_path.open("a", encoding="utf-8") as handle:
                handle.write(run_record.model_dump_json() + "\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_baseline_summary(
        split_name=split_name,
        runs_output_path=runs_output_path,
        summary_path=output_dir / f"{split_name}_baseline_summary.json",
    )
    return records


def _load_textbook_chunks(path: Path) -> list[TextbookChunk]:
    if path.is_dir():
        chunks: list[TextbookChunk] = []
        for jsonl_path in sorted(path.glob("*.jsonl")):
            chunks.extend(read_jsonl(jsonl_path, TextbookChunk))
        return chunks
    return read_jsonl(path, TextbookChunk)


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


def _limit_questions_per_category(
    questions: list[NormalizedQuestion],
    max_per_category: int,
) -> list[NormalizedQuestion]:
    counts: dict[str, int] = {}
    limited: list[NormalizedQuestion] = []
    for question in questions:
        key = question.category.value
        if counts.get(key, 0) >= max_per_category:
            continue
        counts[key] = counts.get(key, 0) + 1
        limited.append(question)
    return limited


def _write_baseline_summary(*, split_name: str, runs_output_path: Path, summary_path: Path) -> None:
    run_records = read_jsonl(runs_output_path, BaselineRunRecord) if runs_output_path.exists() else []
    verdict_counts = Counter(run.report.verdict.value for run in run_records)
    category_counts = Counter(run.spec.category.value for run in run_records)
    score_totals: dict[str, float] = {}
    for run in run_records:
        for key, value in run.evaluation.scores.items():
            score_totals[key] = score_totals.get(key, 0.0) + value

    total = len(run_records)
    rejected = verdict_counts.get("fail", 0) + verdict_counts.get("revise", 0)
    score_means = {
        key: round(total_value / total, 3)
        for key, total_value in sorted(score_totals.items())
        if total
    }
    write_json(
        summary_path,
        {
            "split_name": split_name,
            "total_runs": total,
            "verdict_counts": dict(sorted(verdict_counts.items())),
            "rejected_runs": rejected,
            "rejection_rate": round(rejected / total, 4) if total else None,
            "pass_rate": round(verdict_counts.get("pass", 0) / total, 4) if total else None,
            "category_counts": dict(sorted(category_counts.items())),
            "score_means": score_means,
        },
    )
