from __future__ import annotations

import re

from scibowl.schema.generation import QuestionSpec, RetrievalBundle, RetrievedFactChunk, RetrievedStyleExample
from scibowl.schema.question import NormalizedQuestion
from scibowl.schema.textbook import TextbookChunk
from scibowl.utils.text import lexical_overlap_score


def _query_terms(spec: QuestionSpec) -> set[str]:
    pieces = [spec.subcategory, *spec.topic_focus]
    return set(re.findall(r"[a-z0-9]+", " ".join(pieces).lower()))


def retrieve_bundle(
    spec: QuestionSpec,
    textbook_chunks: list[TextbookChunk],
    style_questions: list[NormalizedQuestion],
    fact_top_k: int = 3,
    style_top_k: int = 3,
) -> RetrievalBundle:
    query_terms = _query_terms(spec)

    fact_hits = sorted(
        textbook_chunks,
        key=lambda chunk: lexical_overlap_score(query_terms, f"{chunk.title} {' '.join(chunk.topics)} {chunk.text}"),
        reverse=True,
    )[:fact_top_k]

    style_hits = [
        question
        for question in style_questions
        if question.category == spec.category and question.question_type == spec.question_type
    ]
    style_hits = sorted(
        style_hits,
        key=lambda question: (
            lexical_overlap_score(query_terms, f"{question.subcategory} {' '.join(question.content_tags)} {question.question_text}"),
            -abs(question.difficulty - spec.difficulty),
        ),
        reverse=True,
    )[:style_top_k]

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
