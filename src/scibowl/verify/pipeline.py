from __future__ import annotations

from scibowl.schema.common import ModelInfo
from scibowl.schema.generation import GeneratedDraft, QuestionSpec, RetrievalBundle
from scibowl.schema.question import NormalizedQuestion
from scibowl.schema.verification import ReviewMetadata, VerificationChecks, VerificationIssue, VerifierReport
from scibowl.utils.ids import make_id
from scibowl.verify.llm_verifier import PromptVerifierModel, build_verifier_model
from scibowl.verify.rules import build_checks, derive_verdict


class VerifierService:
    def __init__(
        self,
        model_name: str = "rule-verifier-v1",
        prompt_version: str = "verifier_v1",
        llm_model: PromptVerifierModel | None = None,
    ) -> None:
        self.llm_model = llm_model or build_verifier_model()
        if self.llm_model is None:
            self.model_info = ModelInfo(provider="local", model_name=model_name, prompt_version=prompt_version)
        else:
            self.model_info = ModelInfo(
                provider="openai_compatible",
                model_name=self.llm_model.model_name,
                prompt_version=self.llm_model.prompt_version,
            )

    def verify(
        self,
        spec: QuestionSpec,
        draft: GeneratedDraft,
        bundle: RetrievalBundle,
        style_questions: list[NormalizedQuestion],
    ) -> VerifierReport:
        checks = build_checks(spec, draft, bundle, style_questions)
        llm_summary: str | None = None
        llm_required_revisions: list[str] = []
        if self.llm_model is not None:
            try:
                review = self.llm_model.review(spec, draft, bundle)
            except Exception:
                review = None
            if review is not None:
                self._merge_llm_review(checks, review)
                llm_summary = str(review.get("summary", "")).strip() or None
                llm_required_revisions = _as_issue_messages(review.get("required_revisions"))
        verdict = derive_verdict(checks)
        required_revisions = []
        for check in [
            checks.format_compliance,
            checks.factual_grounding,
            checks.answerability,
            checks.style_alignment,
        ]:
            required_revisions.extend(issue.message for issue in check.issues)
        required_revisions.extend(item for item in llm_required_revisions if item not in required_revisions)
        return VerifierReport(
            report_id=make_id("verify"),
            draft_id=draft.draft_id,
            spec_id=spec.spec_id,
            verdict=verdict,
            summary=llm_summary or f"Verification completed with verdict '{verdict.value}'.",
            checks=checks,
            required_revisions=required_revisions,
            review_metadata=ReviewMetadata(
                provider=self.model_info.provider,
                model_name=self.model_info.model_name,
                prompt_version=self.model_info.prompt_version,
            ),
        )

    def _merge_llm_review(self, checks: VerificationChecks, review: dict[str, object]) -> None:
        for message in _as_issue_messages(review.get("format_issues")):
            checks.format_compliance.issues.append(VerificationIssue(code="llm_format_issue", message=message))
        for message in _as_issue_messages(review.get("topic_issues")):
            checks.style_alignment.issues.append(VerificationIssue(code="llm_topic_issue", message=message))
        for message in _as_issue_messages(review.get("scientific_accuracy_issues")):
            checks.factual_grounding.issues.append(
                VerificationIssue(code="llm_scientific_accuracy_issue", message=message)
            )

        solver_answer_matches_expected = review.get("solver_answer_matches_expected")
        solver_answer = str(review.get("solver_answer", "")).strip()
        if solver_answer_matches_expected is False:
            mismatch_message = "Verifier-solved answer does not match the provided answer."
            if solver_answer:
                mismatch_message = (
                    f"Verifier-solved answer does not match the provided answer. Solver answer: {solver_answer}"
                )
            checks.factual_grounding.issues.append(
                VerificationIssue(code="solver_answer_mismatch", message=mismatch_message)
            )

        checks.format_compliance.passed = not checks.format_compliance.issues
        checks.factual_grounding.passed = not checks.factual_grounding.issues
        checks.answerability.passed = not checks.answerability.issues
        checks.style_alignment.passed = not checks.style_alignment.issues

        style_score = review.get("style_score")
        if isinstance(style_score, (int, float)):
            checks.style_alignment.style_score = float(style_score)


def _as_issue_messages(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
