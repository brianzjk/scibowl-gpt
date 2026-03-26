from scibowl.prompts.renderer import render_writer_prompt
from scibowl.schema.common import Category, QuestionType
from scibowl.schema.generation import QuestionSpec, RetrievalBundle


def test_render_writer_prompt_includes_biology_specific_guidance() -> None:
    spec = QuestionSpec(
        spec_id="spec_bio_prompt",
        category=Category.BIOLOGY,
        subcategory="genetics",
        question_type=QuestionType.TOSSUP,
        difficulty=4,
        topic_focus=["gene regulation"],
    )

    prompt = render_writer_prompt(spec, RetrievalBundle(spec_id=spec.spec_id))

    assert "Category-Specific Writing Guidance:" in prompt
    assert "Favor mechanistic, causal, comparative, and functional reasoning" in prompt
    assert "in contrast to" in prompt
    assert "State the actual interrogative target early enough" in prompt


def test_render_writer_prompt_uses_category_specific_guidance_for_math() -> None:
    spec = QuestionSpec(
        spec_id="spec_math_prompt",
        category=Category.MATH,
        subcategory="Algebra",
        question_type=QuestionType.TOSSUP,
        difficulty=4,
        topic_focus=["polynomials"],
    )

    prompt = render_writer_prompt(spec, RetrievalBundle(spec_id=spec.spec_id))

    assert "Favor multi-step conceptual setup, recognition of structure, and elegant reasoning" in prompt
    assert "mechanistic, causal, comparative, and functional reasoning" not in prompt
