from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

from scibowl.schema.generation import QuestionSpec, RetrievalBundle, RetrievedFactChunk, RetrievedStyleExample
from scibowl.schema.question import NormalizedQuestion
from scibowl.schema.textbook import TextbookChunk
from scibowl.utils.subcategories import (
    EARTH_SPACE_ASTRO_SUBCATEGORIES,
    EARTH_SPACE_EARTH_SUBCATEGORIES,
    canonicalize_subcategory,
    expand_subcategory_phrases,
)
from scibowl.utils.text import lexical_overlap_score


def _query_terms(spec: QuestionSpec) -> set[str]:
    pieces: list[str] = []
    for item in [spec.subcategory, *spec.topic_focus]:
        pieces.extend(expand_subcategory_phrases(item))
    return set(re.findall(r"[a-z0-9]+", " ".join(pieces).lower()))


@lru_cache(maxsize=1)
def _load_textbook_manifest() -> dict[str, list[str]]:
    path = Path(__file__).resolve().parents[3] / "configs" / "textbook_manifest.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(key): [str(item) for item in value] for key, value in payload.items() if isinstance(value, list)}


def _select_textbook_chunks(spec: QuestionSpec, textbook_chunks: list[TextbookChunk]) -> list[TextbookChunk]:
    if spec.must_use_sources:
        required = {item.strip() for item in spec.must_use_sources if item.strip()}
        return [chunk for chunk in textbook_chunks if chunk.document_id in required]

    manifest = _load_textbook_manifest()
    allowed_ids = manifest.get(spec.category.value)
    if spec.category.value == "earth_space":
        subcategory = canonicalize_subcategory(spec.subcategory)
        if subcategory == "Observation":
            allowed_ids = ["burns_practical_observational_astronomy"]
        elif subcategory in EARTH_SPACE_ASTRO_SUBCATEGORIES:
            allowed_ids = ["seeds_foundations_of_astrophysics"]
        elif subcategory == "Meteorology":
            allowed_ids = ["ahrens_essentials_of_meteorology"]
        elif subcategory == "Hydrology":
            allowed_ids = ["garrison_essentials_of_oceanography_5e"]
        elif subcategory in EARTH_SPACE_EARTH_SUBCATEGORIES:
            allowed_ids = ["tarbuck_earth_science"]
    if allowed_ids is None:
        return textbook_chunks
    if not allowed_ids:
        return []
    return [chunk for chunk in textbook_chunks if chunk.document_id in allowed_ids]


def retrieve_bundle(
    spec: QuestionSpec,
    textbook_chunks: list[TextbookChunk],
    style_questions: list[NormalizedQuestion],
    fact_top_k: int = 3,
    style_top_k: int = 3,
    avoid_fact_chunk_ids: set[str] | None = None,
    avoid_style_question_ids: set[str] | None = None,
) -> RetrievalBundle:
    query_terms = _query_terms(spec)
    relevant_chunks = _select_textbook_chunks(spec, textbook_chunks)
    avoided_fact_ids = avoid_fact_chunk_ids or set()
    avoided_style_ids = avoid_style_question_ids or set()

    fact_candidates = sorted(
        relevant_chunks,
        key=lambda chunk: _bundle_rank_score(
            spec.spec_id,
            chunk.chunk_id,
            lexical_overlap_score(query_terms, f"{chunk.title} {' '.join(chunk.topics)} {chunk.text}")
            - (0.8 if chunk.chunk_id in avoided_fact_ids else 0.0),
        ),
        reverse=True,
    )[: max(fact_top_k * 8, 12)]
    fact_hits = _select_diverse_hits(
        spec.spec_id,
        fact_candidates,
        fact_top_k,
        score_fn=lambda chunk: lexical_overlap_score(query_terms, f"{chunk.title} {' '.join(chunk.topics)} {chunk.text}"),
        text_fn=lambda chunk: f"{chunk.title} {' '.join(chunk.topics)} {chunk.text}",
        id_fn=lambda chunk: chunk.chunk_id,
        diversity_penalty=0.35,
    )

    style_hits = [
        question
        for question in style_questions
        if question.category == spec.category
        and question.question_type == spec.question_type
        and question.question_id not in avoided_style_ids
    ]
    style_candidates = sorted(
        style_hits,
        key=lambda question: (
            _bundle_rank_score(
                spec.spec_id,
                question.question_id,
                lexical_overlap_score(
                    query_terms,
                    f"{question.subcategory} {' '.join(question.content_tags)} {question.question_text}",
                ),
            ),
            -abs(question.difficulty - spec.difficulty),
        ),
        reverse=True,
    )[: max(style_top_k * 8, 18)]
    style_hits = _select_diverse_hits(
        spec.spec_id,
        style_candidates,
        style_top_k,
        score_fn=lambda question: lexical_overlap_score(
            query_terms,
            f"{question.subcategory} {' '.join(question.content_tags)} {question.question_text}",
        ),
        text_fn=lambda question: f"{question.subcategory} {' '.join(question.content_tags)} {question.question_text}",
        id_fn=lambda question: question.question_id,
        diversity_penalty=0.45,
    )

    return RetrievalBundle(
        spec_id=spec.spec_id,
        fact_chunks=[
            RetrievedFactChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                locator=chunk.metadata.get("source_path") if isinstance(chunk.metadata, dict) else None,
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


def _bundle_rank_score(spec_id: str, item_id: str, base_score: float) -> float:
    return base_score + _stable_jitter(spec_id, item_id)


def _stable_jitter(spec_id: str, item_id: str) -> float:
    digest = hashlib.sha256(f"{spec_id}:{item_id}".encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) / 0xFFFFFFFF) * 0.01


def _select_diverse_hits[T](
    spec_id: str,
    candidates: list[T],
    top_k: int,
    *,
    score_fn,
    text_fn,
    id_fn,
    diversity_penalty: float,
) -> list[T]:
    if top_k <= 0 or not candidates:
        return []

    selected: list[T] = []
    remaining = list(candidates)
    while remaining and len(selected) < top_k:
        best_item = max(
            remaining,
            key=lambda item: _mmr_score(
                spec_id,
                id_fn(item),
                score_fn(item),
                text_fn(item),
                [text_fn(existing) for existing in selected],
                diversity_penalty,
            ),
        )
        selected.append(best_item)
        remaining.remove(best_item)
    return selected


def _mmr_score(
    spec_id: str,
    item_id: str,
    base_score: float,
    text: str,
    selected_texts: list[str],
    diversity_penalty: float,
) -> float:
    base = _bundle_rank_score(spec_id, item_id, base_score)
    if not selected_texts:
        return base
    text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    redundancy = max(lexical_overlap_score(text_terms, selected_text) for selected_text in selected_texts)
    return base - diversity_penalty * redundancy
