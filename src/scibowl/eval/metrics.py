from __future__ import annotations

from scibowl.schema.generation import GeneratedDraft, QuestionSpec
from scibowl.schema.verification import VerifierReport


def score_draft(spec: QuestionSpec, draft: GeneratedDraft, report: VerifierReport) -> dict[str, float]:
    style_score = report.checks.style_alignment.style_score or 0.0
    novelty_score = 1.0 - (report.checks.novelty.similarity_score or 0.0)
    factuality = 1.0 if report.checks.factual_grounding.passed else 0.0
    difficulty = 1.0 if report.checks.difficulty_alignment.passed else 0.5
    answer_uniqueness = 1.0 if report.checks.answerability.passed else 0.0
    overall = round((style_score + novelty_score + factuality + difficulty + answer_uniqueness) / 5, 3)
    return {
        "factuality": factuality,
        "style_alignment": round(style_score, 3),
        "difficulty_alignment": difficulty,
        "answer_uniqueness": answer_uniqueness,
        "novelty": round(novelty_score, 3),
        "overall": overall,
    }
