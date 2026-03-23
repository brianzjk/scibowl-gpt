from __future__ import annotations

from pathlib import Path

from scibowl.schema.generation import GeneratedDraft, QuestionSpec, RetrievalBundle


PROMPT_DIR = Path(__file__).resolve().parent / "templates"


def load_template(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def render_writer_prompt(spec: QuestionSpec, bundle: RetrievalBundle) -> str:
    template = load_template("writer_user.txt")
    facts = "\n".join(
        f"- [{chunk.chunk_id}] {chunk.text}"
        for chunk in bundle.fact_chunks
    ) or "- none"
    styles = "\n".join(
        f"- [{example.question_id}] {example.question_text}\n  {example.answer_text}"
        for example in bundle.style_examples
    ) or "- none"
    return template.format(
        category=spec.category.value,
        subcategory=spec.subcategory,
        question_type=spec.question_type.value,
        answer_mode=spec.answer_mode.value,
        difficulty=spec.difficulty,
        topic_focus=", ".join(spec.topic_focus) or "none",
        facts=facts,
        styles=styles,
    )


def render_verifier_prompt(spec: QuestionSpec, draft: GeneratedDraft, bundle: RetrievalBundle) -> str:
    template = load_template("verifier_user.txt")
    facts = "\n".join(
        f"- [{chunk.chunk_id}] {chunk.text}"
        for chunk in bundle.fact_chunks
    ) or "- none"
    citations = "\n".join(
        f"- {citation.chunk_id}"
        for citation in draft.citations
    ) or "- none"
    return template.format(
        category=spec.category.value,
        subcategory=spec.subcategory,
        question_type=spec.question_type.value,
        answer_mode=spec.answer_mode.value,
        difficulty=spec.difficulty,
        question_text=draft.question.question_text,
        answer_text=draft.question.answer_text,
        citations=citations,
        facts=facts,
    )
