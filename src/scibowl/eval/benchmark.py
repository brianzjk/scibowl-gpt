from __future__ import annotations

from scibowl.eval.metrics import score_draft
from scibowl.schema.dataset import EvaluationRecord
from scibowl.schema.generation import GeneratedDraft, QuestionSpec
from scibowl.schema.verification import VerifierReport


def build_evaluation_record(
    run_id: str,
    candidate_question_id: str,
    spec: QuestionSpec,
    draft: GeneratedDraft,
    report: VerifierReport,
) -> EvaluationRecord:
    return EvaluationRecord(
        run_id=run_id,
        candidate_question_id=candidate_question_id,
        spec_id=spec.spec_id,
        scores=score_draft(spec, draft, report),
        human_review={"reviewed": False, "notes": ""},
        system_config={
            "writer_model": draft.model_info.model_name,
            "verifier_model": report.review_metadata.model_name,
        },
    )
