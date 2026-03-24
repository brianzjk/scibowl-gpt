from __future__ import annotations

from scibowl.schema.generation import GeneratedDraft, QuestionSpec
from scibowl.schema.verification import VerifierReport


def score_draft(spec: QuestionSpec, draft: GeneratedDraft, report: VerifierReport) -> dict[str, float]:
    format_score = 1.0 if report.checks.format_compliance.passed else 0.0
    topic_score = report.checks.topic_alignment.style_score or 0.0
    style_score = report.checks.style_alignment.style_score or 0.0
    factuality = 1.0 if report.checks.factual_grounding.passed else 0.0
    overall = round((format_score + factuality + topic_score + style_score) / 4, 3)
    return {
        "format_compliance": format_score,
        "factuality": factuality,
        "topic_alignment": round(topic_score, 3),
        "style_alignment": round(style_score, 3),
        "overall": overall,
    }
