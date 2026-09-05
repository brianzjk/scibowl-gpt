from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from scibowl.schema.dataset import GeneratedQuestionRunRecord
from scibowl.schema.generation import GeneratedDraft, QuestionSpec
from scibowl.schema.review import HumanReview, ReviewRatings
from scibowl.schema.verification import VerifierReport
from scibowl.utils.io import ensure_parent


def dump_json(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


@dataclass
class ReviewableRun:
    draft_id: str
    candidate_question_id: str | None
    spec: QuestionSpec
    draft: GeneratedDraft
    report: VerifierReport
    metadata: dict[str, object] = field(default_factory=dict)


class GeneratedQuestionReviewStore:
    def __init__(self, runs_path: Path, output_path: Path, reviewer_id: str) -> None:
        self.runs_path = runs_path
        self.output_path = output_path
        self.reviewer_id = reviewer_id.strip() or "local_reviewer"
        self._lock = Lock()

        loaded_runs = _load_reviewable_runs(runs_path)
        self._run_order = [run.draft_id for run in loaded_runs]
        self._runs = {run.draft_id: run for run in loaded_runs}
        self._reviews: dict[str, HumanReview] = {}

        if output_path.exists():
            with output_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    review = HumanReview.model_validate_json(line)
                    if review.question_id in self._runs:
                        self._reviews[review.question_id] = review

    def summary(self) -> dict[str, object]:
        reviewed = len(self._reviews)
        verifier_score_reviews = 0
        for review in self._reviews.values():
            if not isinstance(review.metadata, dict):
                continue
            scores = review.metadata.get("human_verifier_scores")
            if isinstance(scores, dict) and any(value is not None for value in scores.values()):
                verifier_score_reviews += 1
        return {
            "reviewer_id": self.reviewer_id,
            "total": len(self._run_order),
            "reviewed": reviewed,
            "remaining": len(self._run_order) - reviewed,
            "verifier_score_reviews": verifier_score_reviews,
        }

    def session_items(self, *, filter_name: str = "unreviewed") -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for draft_id in self._run_order:
            has_review = draft_id in self._reviews
            if filter_name == "unreviewed" and has_review:
                continue
            if filter_name == "reviewed" and not has_review:
                continue
            run = self._runs[draft_id]
            items.append(
                {
                    "draft_id": draft_id,
                    "category": run.spec.category.value,
                    "subcategory": run.spec.subcategory,
                    "difficulty": run.spec.difficulty,
                    "review_status": "reviewed" if has_review else "unreviewed",
                    "source_label": str(run.metadata.get("source_label") or run.metadata.get("job_id") or ""),
                }
            )
        return items

    def question_payload(self, draft_id: str) -> dict[str, object]:
        run = self._runs[draft_id]
        review = self._reviews.get(draft_id)
        return {
            "draft_id": draft_id,
            "candidate_question_id": run.candidate_question_id,
            "spec": run.spec.model_dump(mode="json"),
            "draft": run.draft.model_dump(mode="json"),
            "report": run.report.model_dump(mode="json"),
            "display_scores": _build_display_scores(run.report),
            "metadata": run.metadata,
            "review": review.model_dump(mode="json") if review else None,
        }

    def save_review(
        self,
        *,
        draft_id: str,
        difficulty: int | None,
        quality: float | None,
        comment: str | None,
        verifier_scores: dict[str, float | None] | None,
        verifier_comment: str | None,
    ) -> dict[str, object]:
        with self._lock:
            run = self._runs[draft_id]
            clean_comment = (comment or "").strip() or None
            clean_verifier_comment = (verifier_comment or "").strip() or None
            normalized_verifier_scores = _normalize_verifier_scores(verifier_scores)

            if difficulty is not None and not 1 <= difficulty <= 7:
                raise ValueError("difficulty must be between 1 and 7")
            if quality is not None and quality not in {-1.0, 0.0, 1.0}:
                raise ValueError("quality must be one of -1, 0, 1")

            if (
                difficulty is None
                and quality is None
                and clean_comment is None
                and not any(value is not None for value in normalized_verifier_scores.values())
                and clean_verifier_comment is None
            ):
                self._reviews.pop(draft_id, None)
                self._write_output()
                return self.question_payload(draft_id)

            review = self._reviews.get(draft_id)
            if review is None:
                review = HumanReview(
                    review_id=f"generated_review_{draft_id}",
                    question_id=draft_id,
                    reviewer_id=self.reviewer_id,
                )
            review.ratings = ReviewRatings(quality=quality, difficulty=difficulty)
            review.comment = clean_comment
            review.status = "reviewed"
            review.tournament = str(run.metadata.get("tournament") or run.metadata.get("source_label") or "generated")
            review.metadata = {
                **run.metadata,
                "candidate_question_id": run.candidate_question_id,
                "writer_model": run.draft.model_info.model_name,
                "verifier_model": run.report.review_metadata.model_name,
                "human_verifier_scores": normalized_verifier_scores,
                "verifier_comment": clean_verifier_comment,
            }
            self._reviews[draft_id] = review
            self._write_output()
            return self.question_payload(draft_id)

    def _write_output(self) -> None:
        ensure_parent(self.output_path)
        tmp_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for draft_id in self._run_order:
                review = self._reviews.get(draft_id)
                if review is None:
                    continue
                handle.write(review.model_dump_json() + "\n")
        tmp_path.replace(self.output_path)


def _load_reviewable_runs(path: Path) -> list[ReviewableRun]:
    loaded: list[ReviewableRun] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            loaded.append(_parse_reviewable_run(line))
    return loaded


def _parse_reviewable_run(line: str) -> ReviewableRun:
    run = GeneratedQuestionRunRecord.model_validate_json(line)
    return ReviewableRun(
        draft_id=run.draft.draft_id,
        candidate_question_id=None,
        spec=run.spec,
        draft=run.draft,
        report=run.report,
        metadata={
            **run.metadata,
            "job_id": run.job_id,
            "source_label": run.job_id,
        },
    )


def _normalize_verifier_scores(payload: dict[str, float | None] | None) -> dict[str, float | None]:
    keys = [
        "format_compliance",
        "factuality",
        "topic_alignment",
        "style_alignment",
        "overall",
    ]
    output: dict[str, float | None] = {key: None for key in keys}
    if not isinstance(payload, dict):
        return output
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be between 0.0 and 1.0")
        output[key] = value
    return output


def _build_display_scores(report: VerifierReport) -> dict[str, float]:
    format_score = 1.0 if report.checks.format_compliance.passed else 0.0
    factuality = 1.0 if report.checks.factual_grounding.passed else 0.0
    topic_alignment = round(report.checks.topic_alignment.style_score or 0.0, 3)
    style_alignment = round(report.checks.style_alignment.style_score or 0.0, 3)
    overall = round((format_score + factuality + topic_alignment + style_alignment) / 4, 3)
    return {
        "format_compliance": format_score,
        "factuality": factuality,
        "topic_alignment": topic_alignment,
        "style_alignment": style_alignment,
        "overall": overall,
    }
