from __future__ import annotations

from scibowl.schema.common import Verdict
from scibowl.schema.generation import GeneratedDraft, QuestionSpec, RetrievalBundle
from scibowl.schema.question import NormalizedQuestion
from scibowl.schema.verification import VerificationCheck, VerificationChecks, VerificationIssue


def _format_check(draft: GeneratedDraft) -> VerificationCheck:
    issues: list[VerificationIssue] = []
    if not draft.question.answer_text.startswith("ANSWER:"):
        issues.append(VerificationIssue(code="missing_answer_prefix", message="Answer must start with 'ANSWER:'"))
    return VerificationCheck(passed=not issues, issues=issues)


def _factual_check(draft: GeneratedDraft, bundle: RetrievalBundle) -> VerificationCheck:
    issues: list[VerificationIssue] = []
    if not draft.citations:
        issues.append(VerificationIssue(code="missing_citation", message="Draft has no citations"))
    elif not bundle.fact_chunks:
        issues.append(VerificationIssue(code="missing_fact_chunk", message="No fact chunks were retrieved"))
    return VerificationCheck(passed=not issues, issues=issues)


def _answerability_check(draft: GeneratedDraft) -> VerificationCheck:
    issues: list[VerificationIssue] = []
    if len(draft.question.answer_text.replace("ANSWER:", "").strip()) < 2:
        issues.append(VerificationIssue(code="empty_answer", message="Answer text is too short"))
    if "science bowl question" not in draft.question.question_text.lower():
        issues.append(
            VerificationIssue(
                code="style_generic",
                message="Question wording should be revised toward a Science Bowl question format",
            )
        )
    return VerificationCheck(passed=not issues, issues=issues)


def _difficulty_check(spec: QuestionSpec, draft: GeneratedDraft) -> VerificationCheck:
    estimated = min(7, max(1, len(draft.question.question_text.split()) // 20 + 1))
    issues: list[VerificationIssue] = []
    if abs(estimated - spec.difficulty) > 2:
        issues.append(
            VerificationIssue(
                code="difficulty_mismatch",
                message=f"Estimated difficulty {estimated} is too far from requested {spec.difficulty}",
            )
        )
    return VerificationCheck(passed=not issues, issues=issues, estimated_difficulty=estimated)


def _style_check(spec: QuestionSpec, style_questions: list[NormalizedQuestion]) -> VerificationCheck:
    matched = [
        question
        for question in style_questions
        if question.category == spec.category and question.question_type == spec.question_type
    ]
    score = 1.0 if matched else 0.4
    issues: list[VerificationIssue] = []
    if not matched:
        issues.append(VerificationIssue(code="missing_style_refs", message="No comparable style examples were available"))
    return VerificationCheck(passed=score >= 0.6, issues=issues, style_score=score)


def _novelty_check(draft: GeneratedDraft, style_questions: list[NormalizedQuestion]) -> VerificationCheck:
    draft_terms = set(draft.question.question_text.lower().split())
    max_overlap = 0.0
    for question in style_questions:
        overlap = len(draft_terms & set(question.question_text.lower().split()))
        max_overlap = max(max_overlap, overlap / max(1, len(draft_terms)))
    similarity = round(max_overlap, 3)
    issues: list[VerificationIssue] = []
    if similarity > 0.9:
        issues.append(VerificationIssue(code="too_similar", message="Draft is too close to an existing style example"))
    return VerificationCheck(passed=similarity <= 0.9, issues=issues, similarity_score=similarity)


def build_checks(
    spec: QuestionSpec,
    draft: GeneratedDraft,
    bundle: RetrievalBundle,
    style_questions: list[NormalizedQuestion],
) -> VerificationChecks:
    return VerificationChecks(
        format_compliance=_format_check(draft),
        factual_grounding=_factual_check(draft, bundle),
        answerability=_answerability_check(draft),
        difficulty_alignment=_difficulty_check(spec, draft),
        style_alignment=_style_check(spec, style_questions),
        novelty=_novelty_check(draft, style_questions),
    )


def derive_verdict(checks: VerificationChecks) -> Verdict:
    if all(
        check.passed
        for check in [
            checks.format_compliance,
            checks.factual_grounding,
            checks.answerability,
            checks.difficulty_alignment,
        ]
    ):
        return Verdict.PASS
    if checks.format_compliance.passed and checks.factual_grounding.passed:
        return Verdict.REVISE
    return Verdict.FAIL
