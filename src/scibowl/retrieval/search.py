from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, TypeVar

import yaml

from scibowl.schema.generation import QuestionSpec, RetrievalBundle, RetrievedFactChunk, RetrievedStyleExample
from scibowl.schema.question import NormalizedQuestion
from scibowl.schema.textbook import TextbookChunk
from scibowl.utils.subcategories import (
    EARTH_SPACE_ASTRO_SUBCATEGORIES,
    EARTH_SPACE_EARTH_SUBCATEGORIES,
    canonicalize_subcategory,
    expand_subcategory_phrases,
)


T = TypeVar('T')

_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has',
    'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that', 'the', 'these',
    'this', 'to', 'was', 'were', 'which', 'with',
}


@dataclass(frozen=True)
class _ScoredItem:
    item: object
    score: float


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r'[a-z0-9]+', text.casefold())
        if token not in _STOPWORDS and len(token) > 1
    ]


def _query_weights(spec: QuestionSpec) -> dict[str, float]:
    weights: dict[str, float] = {}

    def add(text: str, weight: float) -> None:
        for token in _tokenize(text):
            weights[token] = max(weights.get(token, 0.0), weight)

    add(spec.subcategory, 1.0)
    for topic in spec.topic_focus:
        add(topic, 3.0)
    direct_terms = set(weights)
    for phrase in expand_subcategory_phrases(spec.subcategory):
        for token in _tokenize(phrase):
            if token not in direct_terms:
                weights[token] = max(weights.get(token, 0.0), 0.25)
    return weights


@lru_cache(maxsize=1)
def _load_textbook_manifest() -> dict[str, list[str]]:
    path = Path(__file__).resolve().parents[3] / 'configs' / 'textbook_manifest.json'
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    return {
        str(key): [str(item) for item in value]
        for key, value in payload.items()
        if isinstance(value, list)
    }


@lru_cache(maxsize=1)
def _load_retrieval_config() -> dict[str, float | int]:
    path = Path(__file__).resolve().parents[3] / 'configs' / 'retrieval.yaml'
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as handle:
        payload = yaml.safe_load(handle) or {}
    return {
        str(key): value
        for key, value in payload.items()
        if isinstance(value, (int, float))
    }


def _candidate_textbook_chunks(
    spec: QuestionSpec,
    textbook_chunks: list[TextbookChunk],
) -> list[TextbookChunk]:
    if spec.must_use_sources:
        required = {item.strip() for item in spec.must_use_sources if item.strip()}
        return _dedupe_chunks(
            [chunk for chunk in textbook_chunks if chunk.document_id in required]
        )

    allowed_ids = _load_textbook_manifest().get(spec.category.value)
    if allowed_ids is None:
        return _dedupe_chunks(textbook_chunks)
    if not allowed_ids:
        return []
    allowed = set(allowed_ids)
    return _dedupe_chunks(
        [chunk for chunk in textbook_chunks if chunk.document_id in allowed]
    )


def _dedupe_chunks(chunks: list[TextbookChunk]) -> list[TextbookChunk]:
    by_id: dict[str, TextbookChunk] = {}
    for chunk in chunks:
        by_id.setdefault(chunk.chunk_id, chunk)
    return [by_id[key] for key in sorted(by_id)]


def _preferred_document_ids(spec: QuestionSpec) -> set[str]:
    if spec.category.value != 'earth_space':
        return set()
    subcategory = canonicalize_subcategory(spec.subcategory)
    if subcategory == 'Observation':
        return {'burns_practical_observational_astronomy'}
    if subcategory in EARTH_SPACE_ASTRO_SUBCATEGORIES:
        return {'seeds_foundations_of_astrophysics'}
    if subcategory == 'Meteorology':
        return {'ahrens_essentials_of_meteorology'}
    if subcategory == 'Hydrology':
        return {'garrison_essentials_of_oceanography_5e'}
    if subcategory in EARTH_SPACE_EARTH_SUBCATEGORIES:
        return {'tarbuck_earth_science'}
    return set()


def _bm25_fact_scores(
    query_weights: dict[str, float],
    chunks: list[TextbookChunk],
) -> dict[str, float]:
    if not query_weights or not chunks:
        return {}
    core_terms = {
        term
        for term, weight in query_weights.items()
        if weight >= 1.0
    }
    topic_terms = {
        term
        for term, weight in query_weights.items()
        if weight >= 3.0
    }
    minimum_topic_matches = 1 if len(topic_terms) <= 2 else min(3, math.ceil(len(topic_terms) / 3))

    term_frequencies: dict[str, Counter[str]] = {}
    document_terms: dict[str, set[str]] = {}
    body_lengths: dict[str, int] = {}
    document_frequency: Counter[str] = Counter()

    for chunk in chunks:
        body_tokens = _tokenize(chunk.text)
        heading_tokens = _tokenize(
            ' '.join(
                [
                    *chunk.topics,
                    chunk.chapter or '',
                    chunk.section or '',
                ]
            )
        )
        counts = Counter(body_tokens)
        for token in heading_tokens:
            counts[token] += 2
        present = set(counts)
        term_frequencies[chunk.chunk_id] = counts
        document_terms[chunk.chunk_id] = present
        body_lengths[chunk.chunk_id] = max(1, len(body_tokens))
        document_frequency.update(present)

    average_length = sum(body_lengths.values()) / len(body_lengths)
    document_count = len(chunks)
    scores: dict[str, float] = {}
    for chunk in chunks:
        counts = term_frequencies[chunk.chunk_id]
        if not core_terms.intersection(document_terms[chunk.chunk_id]):
            scores[chunk.chunk_id] = 0.0
            continue
        if topic_terms and len(topic_terms.intersection(document_terms[chunk.chunk_id])) < minimum_topic_matches:
            scores[chunk.chunk_id] = 0.0
            continue
        length_norm = 0.25 + 0.75 * body_lengths[chunk.chunk_id] / average_length
        score = 0.0
        for term, query_weight in query_weights.items():
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            frequency_docs = document_frequency[term]
            inverse_frequency = math.log(1 + (document_count - frequency_docs + 0.5) / (frequency_docs + 0.5))
            term_score = frequency * 2.2 / (frequency + 1.2 * length_norm)
            score += query_weight * inverse_frequency * term_score
        scores[chunk.chunk_id] = score
    return scores


def retrieve_bundle(
    spec: QuestionSpec,
    textbook_chunks: list[TextbookChunk],
    style_questions: list[NormalizedQuestion],
    fact_top_k: int | None = None,
    style_top_k: int | None = None,
    avoid_fact_chunk_ids: set[str] | None = None,
    avoid_style_question_ids: set[str] | None = None,
) -> RetrievalBundle:
    config = _load_retrieval_config()
    fact_limit = fact_top_k if fact_top_k is not None else int(config.get('fact_top_k', 3))
    style_limit = style_top_k if style_top_k is not None else int(config.get('style_top_k', 3))
    minimum_fact_score = float(config.get('minimum_fact_score', 0.05))
    preferred_boost = float(config.get('preferred_document_boost', 0.2))
    fact_diversity = float(config.get('fact_diversity_penalty', 0.25))
    style_diversity = float(config.get('style_diversity_penalty', 0.25))

    avoided_fact_ids = avoid_fact_chunk_ids or set()
    candidate_chunks = [
        chunk
        for chunk in _candidate_textbook_chunks(spec, textbook_chunks)
        if chunk.chunk_id not in avoided_fact_ids
    ]
    query_weights = _query_weights(spec)
    fact_scores = _bm25_fact_scores(query_weights, candidate_chunks)
    preferred_ids = _preferred_document_ids(spec)
    scored_facts: list[_ScoredItem] = []
    for chunk in candidate_chunks:
        raw_score = fact_scores.get(chunk.chunk_id, 0.0)
        if raw_score < minimum_fact_score:
            continue
        score = raw_score + (preferred_boost if chunk.document_id in preferred_ids else 0.0)
        scored_facts.append(_ScoredItem(item=chunk, score=score))
    fact_hits = _select_diverse_hits(
        scored_facts,
        fact_limit,
        text_fn=lambda item: item.text,
        id_fn=lambda item: item.chunk_id,
        diversity_penalty=fact_diversity,
    )

    avoided_style_ids = avoid_style_question_ids or set()
    style_candidates = [
        question
        for question in style_questions
        if question.category == spec.category
        and question.question_type == spec.question_type
        and question.question_id not in avoided_style_ids
    ]
    targeted_ids = set(spec.style_target_ids)
    scored_styles = [
        _ScoredItem(
            item=question,
            score=_style_score(spec, question, targeted_ids),
        )
        for question in style_candidates
    ]
    style_hits = _select_diverse_hits(
        scored_styles,
        style_limit,
        text_fn=lambda item: item.question_text,
        id_fn=lambda item: item.question_id,
        diversity_penalty=style_diversity,
    )

    return RetrievalBundle(
        spec_id=spec.spec_id,
        fact_chunks=[
            RetrievedFactChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                locator=_chunk_locator(chunk),
                text=chunk.text,
            )
            for chunk in fact_hits
        ],
        style_examples=[
            RetrievedStyleExample(
                question_id=question.question_id,
                question_text=question.question_text,
                answer_text=question.answer_text,
                difficulty=question.difficulty,
                question_type=question.question_type,
                answer_mode=question.answer_mode,
            )
            for question in style_hits
        ],
    )


def _style_score(
    spec: QuestionSpec,
    question: NormalizedQuestion,
    targeted_ids: set[str],
) -> float:
    score = 100.0 if question.question_id in targeted_ids else 0.0
    tournament = question.source_metadata.tournament or ''
    year = question.source_metadata.year or _year_from_text(
        f'{question.source_id} {tournament}'
    )
    is_mit = question.source_id.casefold().startswith('mit_') or 'mit science bowl' in (
        question.source_metadata.tournament or ''
    ).casefold()
    if is_mit:
        score += 1.2 if year is not None and year >= 2024 else 0.5
    elif question.source_id.casefold().startswith('nsb_'):
        score += 0.25

    quality = question.source_metadata.original_quality
    if quality is not None:
        score += 0.4 * max(-1.0, min(1.0, quality))
    rated_difficulty = question.source_metadata.original_difficulty
    if rated_difficulty is not None:
        score -= 0.25 * abs(rated_difficulty - spec.difficulty)
    if spec.answer_mode is not None and question.answer_mode == spec.answer_mode:
        score += 0.15
    if canonicalize_subcategory(question.subcategory) == canonicalize_subcategory(spec.subcategory):
        score += 0.2
    score += min(0.15, 0.03 * len(question.style_tags))

    topic_terms = set(_query_weights(spec))
    question_terms = set(_tokenize(question.question_text))
    if topic_terms:
        score -= 0.8 * len(topic_terms.intersection(question_terms)) / len(topic_terms)
    return score


def _year_from_text(value: str) -> int | None:
    match = re.search(r'(?<!\d)(20\d{2})(?!\d)', value)
    return int(match.group(1)) if match else None


def _chunk_locator(chunk: TextbookChunk) -> str:
    parts = [chunk.title]
    if chunk.chapter:
        parts.append(chunk.chapter)
    if chunk.section and chunk.section != chunk.chapter:
        parts.append(chunk.section)
    if chunk.pages:
        first_page = min(chunk.pages)
        last_page = max(chunk.pages)
        page_text = f'p. {first_page}' if first_page == last_page else f'pp. {first_page}-{last_page}'
        parts.append(page_text)
    return '; '.join(parts)


def _select_diverse_hits(
    candidates: list[_ScoredItem],
    top_k: int,
    *,
    text_fn: Callable[[T], str],
    id_fn: Callable[[T], str],
    diversity_penalty: float,
) -> list[T]:
    if top_k <= 0 or not candidates:
        return []
    remaining = sorted(
        candidates,
        key=lambda candidate: (-candidate.score, id_fn(candidate.item)),
    )
    selected: list[T] = []
    while remaining and len(selected) < top_k:
        best_index = 0
        best_score = -math.inf
        for index, candidate in enumerate(remaining):
            item = candidate.item
            redundancy = max(
                (_jaccard_similarity(text_fn(item), text_fn(existing)) for existing in selected),
                default=0.0,
            )
            mmr_score = candidate.score - diversity_penalty * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = index
        selected.append(remaining.pop(best_index).item)
    return selected


def _jaccard_similarity(left: str, right: str) -> float:
    left_terms = set(_tokenize(left))
    right_terms = set(_tokenize(right))
    union = left_terms.union(right_terms)
    if not union:
        return 0.0
    return len(left_terms.intersection(right_terms)) / len(union)
