from scibowl.generate.orchestration import GenerationOrchestrator
from scibowl.schema.common import AnswerMode, Category, QuestionType, SourceType
from scibowl.schema.generation import QuestionSpec
from scibowl.schema.question import NormalizedQuestion
from scibowl.schema.textbook import TextbookChunk


def test_demo_pipeline_runs() -> None:
    spec = QuestionSpec(
        spec_id="spec_1",
        category=Category.EARTH_SPACE,
        subcategory="stars",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
        topic_focus=["Balmer lines"],
    )
    chunks = [
        TextbookChunk(
            chunk_id="chunk1",
            document_id="astro",
            title="Astronomy",
            topics=["stars", "spectra"],
            text="A-type stars show strong Balmer absorption lines in their spectra.",
            char_count=65,
            token_count_est=11,
            metadata={},
        )
    ]
    styles = [
        NormalizedQuestion(
            question_id="q1",
            source_type=SourceType.DATASET,
            source_id="style",
            category=Category.EARTH_SPACE,
            subcategory="stars",
            question_type=QuestionType.TOSSUP,
            answer_mode=AnswerMode.SHORT_ANSWER,
            difficulty=4,
            question_text="Which class of stars shows the strongest Balmer lines?",
            answer_text="ANSWER: A-type stars",
        )
    ]

    bundle, draft, report = GenerationOrchestrator().run(spec, chunks, styles)

    assert bundle.fact_chunks
    assert draft.question.answer_text.startswith("ANSWER:")
    assert report.spec_id == spec.spec_id
