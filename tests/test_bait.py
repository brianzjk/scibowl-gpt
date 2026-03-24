from scibowl.schema.common import AnswerMode, Category, Citation, ModelInfo, QuestionType, SourceType, Verdict
from scibowl.schema.generation import DraftQuestion, GeneratedDraft, QuestionSpec, RetrievalBundle, RetrievedFactChunk
from scibowl.schema.question import NormalizedQuestion
from scibowl.verify.bait import sample_answer_prefixes
from scibowl.verify.pipeline import VerifierService
from scibowl.verify.rules import build_checks


def _build_spec() -> QuestionSpec:
    return QuestionSpec(
        spec_id="spec_bait",
        category=Category.CHEMISTRY,
        subcategory="organic chemistry",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=2,
        topic_focus=["hydrocarbons"],
    )


def _build_style_questions() -> list[NormalizedQuestion]:
    return [
        NormalizedQuestion(
            question_id="style_pentane",
            source_type=SourceType.DATASET,
            source_id="style",
            category=Category.CHEMISTRY,
            subcategory="organic chemistry",
            question_type=QuestionType.TOSSUP,
            answer_mode=AnswerMode.SHORT_ANSWER,
            difficulty=2,
            question_text="What straight-chain alkane contains five carbon atoms?",
            answer_text="ANSWER: pentane",
        ),
        NormalizedQuestion(
            question_id="style_cyclohexane",
            source_type=SourceType.DATASET,
            source_id="style",
            category=Category.CHEMISTRY,
            subcategory="organic chemistry",
            question_type=QuestionType.TOSSUP,
            answer_mode=AnswerMode.SHORT_ANSWER,
            difficulty=2,
            question_text="What cyclic saturated hydrocarbon contains six carbon atoms in a ring?",
            answer_text="ANSWER: cyclohexane",
        ),
    ]


def _build_bundle() -> RetrievalBundle:
    return RetrievalBundle(
        spec_id="spec_bait",
        fact_chunks=[
            RetrievedFactChunk(
                chunk_id="fact_pentane",
                document_id="chem_text",
                locator="sec 1",
                text="Pentane is a straight-chain alkane containing five carbon atoms.",
            ),
            RetrievedFactChunk(
                chunk_id="fact_cyclohexane",
                document_id="chem_text",
                locator="sec 2",
                text="Cyclohexane is a cyclic saturated hydrocarbon containing six carbon atoms in a ring.",
            ),
        ],
    )


def _build_draft(question_text: str, answer_text: str = "ANSWER: cyclohexane") -> GeneratedDraft:
    spec = _build_spec()
    return GeneratedDraft(
        draft_id="draft_bait",
        spec_id=spec.spec_id,
        model_info=ModelInfo(provider="test", model_name="test", prompt_version="test"),
        question=DraftQuestion(
            category=spec.category,
            subcategory=spec.subcategory,
            question_type=spec.question_type,
            answer_mode=spec.answer_mode,
            difficulty=spec.difficulty,
            question_text=question_text,
            answer_text=answer_text,
            choices=[],
        ),
        citations=[Citation(source_id="chem_text", chunk_id="fact_cyclohexane", locator="sec 2")],
    )


def test_top_k_sampling_shows_answer_prefix_flip_for_bait_question() -> None:
    draft = _build_draft(
        "What straight-chain alkane contains five carbon atoms, but what cyclic saturated hydrocarbon contains six carbon atoms in a ring in this Science Bowl question?"
    )

    samples = sample_answer_prefixes(draft, _build_bundle(), _build_style_questions(), top_k=2, step_tokens=5)

    top_answers = [sample["top_k"][0]["answer_prefix"] for sample in samples]
    assert "pentane" in top_answers
    assert "cyclohexane" in top_answers


def test_bait_detection_fails_question_with_strong_prefix_switch() -> None:
    spec = _build_spec()
    draft = _build_draft(
        "What straight-chain alkane contains five carbon atoms, but what cyclic saturated hydrocarbon contains six carbon atoms in a ring in this Science Bowl question?"
    )

    checks = build_checks(spec, draft, _build_bundle(), _build_style_questions())

    assert not checks.bait_detection.passed
    assert checks.bait_detection.bait_score and checks.bait_detection.bait_score > 0.0
    assert checks.bait_detection.issues[0].code == "bait_detected"


def test_non_bait_question_passes_bait_detection_and_verifier() -> None:
    spec = _build_spec()
    draft = _build_draft(
        "What cyclic saturated hydrocarbon contains six carbon atoms in a ring in this Science Bowl question?"
    )

    report = VerifierService().verify(spec, draft, _build_bundle(), _build_style_questions())

    assert report.checks.bait_detection.passed
    assert report.verdict == Verdict.PASS
