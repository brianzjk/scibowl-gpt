from __future__ import annotations

import re
from dataclasses import dataclass

from scibowl.schema.generation import GeneratedDraft, RetrievalBundle
from scibowl.schema.question import NormalizedQuestion
from scibowl.schema.verification import VerificationCheck, VerificationIssue
from scibowl.utils.text import lexical_overlap_score, normalize_whitespace


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "which",
    "with",
}

CONTRAST_MARKERS = {"but", "however", "instead", "rather", "except", "although", "yet", "unlike"}


@dataclass
class _AnswerCandidate:
    answer_prefix: str
    evidence_texts: list[str]


def build_bait_check(
    draft: GeneratedDraft,
    bundle: RetrievalBundle,
    style_questions: list[NormalizedQuestion],
    *,
    top_k: int = 3,
) -> VerificationCheck:
    samples = sample_answer_prefixes(draft, bundle, style_questions, top_k=top_k)
    final_prefix = _answer_prefix(draft.question.answer_text)
    significant_samples = [sample for sample in samples if _is_significant(sample)]

    issues: list[VerificationIssue] = []
    bait_score = 0.0
    if final_prefix and significant_samples:
        final_sample = next(
            (sample for sample in reversed(significant_samples) if sample["top_k"][0]["answer_prefix"] == final_prefix),
            None,
        )
        if final_sample is not None:
            misleading_sample = next(
                (
                    sample
                    for sample in significant_samples
                    if sample["top_k"][0]["answer_prefix"] != final_prefix
                    and sample["position"] < final_sample["position"]
                ),
                None,
            )
            if misleading_sample is not None:
                bait_score = round(
                    min(
                        1.0,
                        misleading_sample["top_k"][0]["score"] * final_sample["top_k"][0]["score"],
                    ),
                    3,
                )
                issues.append(
                    VerificationIssue(
                        code="bait_detected",
                        message=(
                            "Top-k answer-prefix sampling found a strong early guess of "
                            f"'{misleading_sample['top_k'][0]['answer_prefix']}' before the question pivots to "
                            f"'{final_prefix}'."
                        ),
                        span=misleading_sample["question_prefix"],
                    )
                )

    return VerificationCheck(
        passed=not issues,
        issues=issues,
        bait_score=bait_score,
        sampled_prefixes=samples,
    )


def sample_answer_prefixes(
    draft: GeneratedDraft,
    bundle: RetrievalBundle,
    style_questions: list[NormalizedQuestion],
    *,
    top_k: int = 3,
    min_prefix_tokens: int = 4,
    step_tokens: int = 4,
) -> list[dict[str, object]]:
    question_tokens = _tokenize(draft.question.question_text)
    if len(question_tokens) < min_prefix_tokens:
        return []

    candidates = _build_candidates(draft, bundle, style_questions)
    if len(candidates) < 2:
        return []

    sample_points = {min(len(question_tokens), token_count) for token_count in range(min_prefix_tokens, len(question_tokens) + 1, step_tokens)}
    sample_points.add(len(question_tokens))

    samples: list[dict[str, object]] = []
    for token_count in sorted(sample_points):
        prefix_tokens = question_tokens[:token_count]
        prefix_text = " ".join(prefix_tokens)
        ranked = _rank_candidates(prefix_tokens, candidates, top_k=top_k)
        if not ranked:
            continue
        samples.append(
            {
                "position": round(token_count / len(question_tokens), 3),
                "question_prefix": prefix_text,
                "top_k": ranked,
            }
        )
    return samples


def _build_candidates(
    draft: GeneratedDraft,
    bundle: RetrievalBundle,
    style_questions: list[NormalizedQuestion],
) -> list[_AnswerCandidate]:
    candidate_map: dict[str, list[str]] = {}

    def add(answer_text: str, evidence_text: str) -> None:
        prefix = _answer_prefix(answer_text)
        if not prefix:
            return
        evidence = normalize_whitespace(evidence_text)
        if not evidence:
            return
        candidate_map.setdefault(prefix, [])
        if evidence not in candidate_map[prefix]:
            candidate_map[prefix].append(evidence)

    for question in style_questions:
        add(question.answer_text, f"{question.question_text} {question.answer_text}")

    cited_chunk_ids = {citation.chunk_id for citation in draft.citations}
    supporting_chunks = [
        chunk for chunk in bundle.fact_chunks if not cited_chunk_ids or chunk.chunk_id in cited_chunk_ids
    ]

    for chunk in supporting_chunks:
        support_text = " ".join(part for part in [chunk.text, chunk.document_id, chunk.locator or ""] if part)
        add(draft.question.answer_text, support_text)

    for chunk in bundle.fact_chunks:
        support_text = " ".join(part for part in [chunk.text, chunk.document_id, chunk.locator or ""] if part)
        for phrase in _extract_candidate_phrases(chunk.text):
            add(phrase, support_text)

    return [
        _AnswerCandidate(answer_prefix=prefix, evidence_texts=evidence_texts)
        for prefix, evidence_texts in candidate_map.items()
    ]


def _extract_candidate_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    patterns = [
        r"\b([A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,3})\s+(?:is|are|was|were)\b",
        r"\b(?:called|known as|named|termed)\s+([A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,3})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            phrase = normalize_whitespace(match.group(1))
            if phrase and phrase.lower() not in STOPWORDS and phrase not in phrases:
                phrases.append(phrase)
    return phrases


def _rank_candidates(
    prefix_tokens: list[str],
    candidates: list[_AnswerCandidate],
    *,
    top_k: int,
) -> list[dict[str, object]]:
    prefix_terms = {token for token in prefix_tokens if token not in STOPWORDS}
    if not prefix_terms:
        return []

    recent_terms = set(prefix_tokens[-4:])
    scored: list[tuple[str, float]] = []
    for candidate in candidates:
        score = 0.0
        for evidence_text in candidate.evidence_texts:
            evidence_terms = set(_tokenize(evidence_text))
            overlap = lexical_overlap_score(prefix_terms, evidence_text)
            recent_overlap = 0.0 if not recent_terms else len(recent_terms & evidence_terms) / len(recent_terms)
            contrast_bonus = 0.2 if CONTRAST_MARKERS & recent_terms else 0.0
            score = max(score, (0.65 * overlap) + (0.25 * recent_overlap) + contrast_bonus)
        if score > 0.0:
            scored.append((candidate.answer_prefix, score))

    if not scored:
        return []

    scored.sort(key=lambda item: (-item[1], item[0]))
    top_scored = scored[:top_k]
    total = sum(score for _, score in top_scored) or 1.0
    return [
        {
            "answer_prefix": answer_prefix,
            "score": round(score / total, 3),
        }
        for answer_prefix, score in top_scored
    ]


def _is_significant(sample: dict[str, object]) -> bool:
    top_k = sample.get("top_k")
    if not isinstance(top_k, list) or not top_k:
        return False
    first_score = float(top_k[0]["score"])
    second_score = float(top_k[1]["score"]) if len(top_k) > 1 else 0.0
    return first_score >= 0.55 and (first_score - second_score) >= 0.15


def _answer_prefix(answer_text: str, max_tokens: int = 3) -> str:
    cleaned = re.sub(r"^ANSWER:\s*", "", answer_text.strip(), flags=re.IGNORECASE)
    tokens = _tokenize(cleaned)
    return " ".join(tokens[:max_tokens])


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
