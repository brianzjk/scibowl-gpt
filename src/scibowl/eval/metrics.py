from __future__ import annotations

from scibowl.schema.generation import GeneratedDraft, QuestionSpec
from scibowl.schema.verification import VerifierReport


def score_draft(spec: QuestionSpec, draft: GeneratedDraft, report: VerifierReport) -> dict[str, float]:
    format_score = 1.0 if report.checks.format_compliance.passed else 0.0
    style_score = report.checks.style_alignment.style_score or 0.0
    factuality = 1.0 if report.checks.factual_grounding.passed else 0.0
    overall = round((format_score + style_score + factuality) / 3, 3)
    return {
        "format_compliance": format_score,
        "factuality": factuality,
        "style_alignment": round(style_score, 3),
        "overall": overall,
    }
