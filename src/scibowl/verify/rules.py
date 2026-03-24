from __future__ import annotations

import re

from scibowl.schema.common import Verdict
from scibowl.schema.generation import GeneratedDraft, QuestionSpec, RetrievalBundle
from scibowl.schema.question import NormalizedQuestion
from scibowl.schema.verification import VerificationCheck, VerificationChecks, VerificationIssue
from scibowl.utils.subcategories import expand_subcategory_phrases
from scibowl.utils.text import lexical_overlap_score


def _format_check(draft: GeneratedDraft) -> VerificationCheck:
    issues: list[VerificationIssue] = []
    question_text = draft.question.question_text.strip()
    lowered = question_text.lower()
    if not draft.question.answer_text.startswith("ANSWER:"):
        issues.append(VerificationIssue(code="missing_answer_prefix", message="Answer must start with 'ANSWER:'"))
    if not question_text:
        issues.append(VerificationIssue(code="empty_question", message="Question text is empty"))
    if draft.question.answer_mode.value == "multiple_choice":
        labels = [choice.label.strip().upper() for choice in draft.question.choices]
        if labels != ["W", "X", "Y", "Z"]:
            issues.append(
                VerificationIssue(
                    code="multiple_choice_format",
                    message="Multiple-choice questions must provide exactly four choices labeled W, X, Y, Z",
                )
            )
        if any(not choice.text.strip() for choice in draft.question.choices):
            issues.append(VerificationIssue(code="blank_choice", message="Multiple-choice answers must not be blank"))
        if any(token in lowered for token in ("w)", "x)", "y)", "z)")):
            issues.append(
                VerificationIssue(
                    code="choices_embedded_in_question",
                    message="Multiple-choice options should be stored in choices, not embedded in question_text",
                )
            )
        if "which of the following" not in lowered:
            issues.append(
                VerificationIssue(
                    code="missing_which_of_the_following",
                    message='Multiple-choice questions must explicitly contain the phrase "Which of the following".',
                )
            )
    elif draft.question.choices:
        issues.append(
            VerificationIssue(
                code="short_answer_has_choices",
                message="Short-answer questions must not include multiple-choice options",
            )
        )
    if not question_text.endswith("?") and not lowered.startswith(("identify all", "rank the following")):
        issues.append(
            VerificationIssue(
                code="non_interrogative_format",
                message="Questions should usually be written as full interrogative sentences",
            )
        )
    return VerificationCheck(passed=not issues, issues=issues)


def _factual_check(draft: GeneratedDraft, bundle: RetrievalBundle) -> VerificationCheck:
    issues: list[VerificationIssue] = []
    if not draft.citations:
        issues.append(VerificationIssue(code="missing_citation", message="Draft has no citations"))
    elif not bundle.fact_chunks:
        issues.append(VerificationIssue(code="missing_fact_chunk", message="No fact chunks were retrieved"))
    suspicious_markers = (
        "table of contents",
        "companion website",
        "mastering",
        "launchpad",
        "instructor resource",
        "online resource",
        "supplement",
        "preface",
        "about the author",
        "glossary",
        "index",
        "answer key",
        "selected answers",
        "references",
    )
    suspicious_chunk_ids = [
        chunk.chunk_id
        for chunk in bundle.fact_chunks
        if any(marker in chunk.text.lower() for marker in suspicious_markers)
    ]
    if suspicious_chunk_ids:
        issues.append(
            VerificationIssue(
                code="suspicious_source_chunk",
                message="Retrieved fact chunks look like front matter or supplemental material rather than textbook content",
                citations_checked=suspicious_chunk_ids,
            )
        )
    return VerificationCheck(passed=not issues, issues=issues)


def _answerability_check(draft: GeneratedDraft) -> VerificationCheck:
    issues: list[VerificationIssue] = []
    lowered = draft.question.question_text.lower()
    answer_core = draft.question.answer_text.replace("ANSWER:", "").strip()
    answer_word_count = len(re.findall(r"\S+", answer_core))
    if len(draft.question.answer_text.replace("ANSWER:", "").strip()) < 2:
        issues.append(VerificationIssue(code="empty_answer", message="Answer text is too short"))
    if draft.question.answer_mode.value == "short_answer":
        lower_answer = answer_core.lower()
        looks_like_sentence = (
            any(mark in answer_core for mark in ".?!;")
            or answer_word_count > 12
            or (
                answer_word_count > 8
                and re.match(r"^(because|it|they|this|that|these|those|the)\b", lower_answer) is not None
            )
        )
        if looks_like_sentence:
            issues.append(
                VerificationIssue(
                    code="sentence_answer_for_short_answer",
                    message=(
                        "Short-answer questions should usually have a concise answer line, not a full sentence or "
                        "explanatory clause. If sentence-length explanation is required, rewrite the question or "
                        "use multiple choice."
                    ),
                )
            )
    textbook_meta_markers = (
        "according to the textbook",
        "according to this textbook",
        "in the textbook",
        "which chapter",
        "what chapter",
        "which page",
        "what page",
        "who wrote this textbook",
        "author of this textbook",
        "authors of this textbook",
    )
    if any(marker in lowered for marker in textbook_meta_markers):
        issues.append(
            VerificationIssue(
                code="source_material_meta",
                message="Questions should use source material for scientific facts, not ask about the source material itself",
            )
        )
    return VerificationCheck(passed=not issues, issues=issues)


def _difficulty_check(spec: QuestionSpec, draft: GeneratedDraft) -> VerificationCheck:
    return VerificationCheck(passed=True, issues=[])


def _topic_check(spec: QuestionSpec, draft: GeneratedDraft) -> VerificationCheck:
    pieces: list[str] = []
    for item in [spec.subcategory, *spec.topic_focus]:
        pieces.extend(expand_subcategory_phrases(item))
    query_terms = set(re.findall(r"[a-z0-9]+", " ".join(pieces).lower()))
    overlap = lexical_overlap_score(query_terms, draft.question.question_text)
    issues: list[VerificationIssue] = []
    if overlap == 0.0:
        issues.append(
            VerificationIssue(
                code="topic_mismatch",
                message="Question text does not appear to follow the requested subcategory/topic focus",
            )
        )
    return VerificationCheck(passed=not issues, issues=issues, style_score=round(min(1.0, overlap * 2.0), 3))


def _style_check(spec: QuestionSpec, draft: GeneratedDraft, style_questions: list[NormalizedQuestion]) -> VerificationCheck:
    issues: list[VerificationIssue] = []
    question_text = draft.question.question_text.strip()
    lowered = question_text.lower()
    word_count = len(question_text.split())
    direct_recall_starts = (
        "what is",
        "what are",
        "which is",
        "which are",
        "name the",
        "identify the",
        "give the",
    )
    conceptual_markers = (
        "why",
        "how",
        "best explains",
        "best describes",
        "would happen",
        "most likely",
        "compared",
        "relative to",
        "because",
        "infer",
        "conclude",
        "predict",
    )

    if lowered.startswith(direct_recall_starts) and word_count <= 18 and not any(marker in lowered for marker in conceptual_markers):
        issues.append(
            VerificationIssue(
                code="reaction_speed_test",
                message="Question reads like a reaction-speed test rather than a conceptual Science Bowl clue.",
            )
        )
    elif lowered.startswith(direct_recall_starts) and not any(marker in lowered for marker in conceptual_markers):
        issues.append(
            VerificationIssue(
                code="overly_direct_recall",
                message="Question is too straightforward and reads like direct fact recall rather than conceptual reasoning.",
            )
        )

    normalized_answer = draft.question.answer_text.replace("ANSWER:", "").strip().lower()
    if normalized_answer and normalized_answer in lowered:
        issues.append(
            VerificationIssue(
                code="answer_in_stem",
                message="The answer or an obvious near-copy of it appears directly in the question text.",
            )
        )

    sentence_count = len([part for part in re.split(r"[.!?]+", question_text) if part.strip()])
    if sentence_count >= 3:
        issues.append(
            VerificationIssue(
                code="unnecessary_flavor_text",
                message="Question contains extra flavor text instead of staying crisp and focused on the core idea.",
            )
        )
    matched = any(question.category == spec.category and question.question_type == spec.question_type for question in style_questions)
    score = 1.0
    if not matched:
        score -= 0.1
    score -= 0.35 * len(issues)
    return VerificationCheck(passed=not issues, issues=issues, style_score=round(max(0.0, score), 3))


def _novelty_check(draft: GeneratedDraft, style_questions: list[NormalizedQuestion]) -> VerificationCheck:
    return VerificationCheck(passed=True, issues=[])


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
        topic_alignment=_topic_check(spec, draft),
        style_alignment=_style_check(spec, draft, style_questions),
        novelty=_novelty_check(draft, style_questions),
    )


def derive_verdict(checks: VerificationChecks) -> Verdict:
    if all(
        check.passed
        for check in [
            checks.format_compliance,
            checks.factual_grounding,
            checks.answerability,
            checks.topic_alignment,
            checks.style_alignment,
        ]
    ):
        return Verdict.PASS
    if checks.factual_grounding.passed:
        return Verdict.REVISE
    return Verdict.FAIL
