from scibowl.schema.common import AnswerMode, Category, Citation, ModelInfo, QuestionType, SourceType, Verdict
from scibowl.schema.generation import DraftQuestion, GeneratedDraft, QuestionSpec, RetrievalBundle, RetrievedFactChunk
from scibowl.schema.question import Choice, NormalizedQuestion
from scibowl.verify.pipeline import VerifierService
from scibowl.verify.rules import build_checks


def _spec(
    *,
    spec_id: str = "spec_default",
    category: Category = Category.CHEMISTRY,
    subcategory: str = "organic chemistry",
    question_type: QuestionType = QuestionType.TOSSUP,
    answer_mode: AnswerMode = AnswerMode.SHORT_ANSWER,
    difficulty: int = 2,
    topic_focus: list[str] | None = None,
) -> QuestionSpec:
    return QuestionSpec(
        spec_id=spec_id,
        category=category,
        subcategory=subcategory,
        question_type=question_type,
        answer_mode=answer_mode,
        difficulty=difficulty,
        topic_focus=topic_focus or ["core fact"],
    )


def _style_question(
    *,
    question_id: str,
    category: Category,
    subcategory: str,
    question_type: QuestionType,
    answer_mode: AnswerMode,
    difficulty: int,
    question_text: str,
    answer_text: str,
) -> NormalizedQuestion:
    return NormalizedQuestion(
        question_id=question_id,
        source_type=SourceType.DATASET,
        source_id="style",
        category=category,
        subcategory=subcategory,
        question_type=question_type,
        answer_mode=answer_mode,
        difficulty=difficulty,
        question_text=question_text,
        answer_text=answer_text,
    )


def _bundle(spec: QuestionSpec, *chunks: tuple[str, str, str]) -> RetrievalBundle:
    return RetrievalBundle(
        spec_id=spec.spec_id,
        fact_chunks=[
            RetrievedFactChunk(chunk_id=chunk_id, document_id=document_id, locator=locator, text=text)
            for chunk_id, document_id, locator, text in chunks
        ],
    )


def _draft(
    spec: QuestionSpec,
    *,
    question_text: str,
    answer_text: str,
    cited_chunk_ids: list[str] | None = None,
    choices: list[Choice] | None = None,
) -> GeneratedDraft:
    cited_chunk_ids = cited_chunk_ids or []
    return GeneratedDraft(
        draft_id=f"draft_{spec.spec_id}",
        spec_id=spec.spec_id,
        model_info=ModelInfo(provider="test", model_name="unit-test", prompt_version="unit"),
        question=DraftQuestion(
            category=spec.category,
            subcategory=spec.subcategory,
            question_type=spec.question_type,
            answer_mode=spec.answer_mode,
            difficulty=spec.difficulty,
            question_text=question_text,
            answer_text=answer_text,
            choices=choices or [],
        ),
        citations=[
            Citation(source_id="textbook", chunk_id=chunk_id, locator="unit-test")
            for chunk_id in cited_chunk_ids
        ],
    )


def _chemistry_styles() -> list[NormalizedQuestion]:
    return [
        _style_question(
            question_id="chem_1",
            category=Category.CHEMISTRY,
            subcategory="organic chemistry",
            question_type=QuestionType.TOSSUP,
            answer_mode=AnswerMode.SHORT_ANSWER,
            difficulty=2,
            question_text="What cyclic saturated hydrocarbon contains six carbon atoms in a ring in this Science Bowl question?",
            answer_text="ANSWER: cyclohexane",
        ),
        _style_question(
            question_id="chem_2",
            category=Category.CHEMISTRY,
            subcategory="organic chemistry",
            question_type=QuestionType.TOSSUP,
            answer_mode=AnswerMode.SHORT_ANSWER,
            difficulty=2,
            question_text="What straight-chain alkane contains five carbon atoms in this Science Bowl question?",
            answer_text="ANSWER: pentane",
        ),
        _style_question(
            question_id="chem_bonus",
            category=Category.CHEMISTRY,
            subcategory="general chemistry",
            question_type=QuestionType.BONUS,
            answer_mode=AnswerMode.MULTIPLE_CHOICE,
            difficulty=3,
            question_text="In this Science Bowl question, which gas is noble at room temperature?",
            answer_text="ANSWER: X) neon",
        ),
    ]


def _biology_styles() -> list[NormalizedQuestion]:
    return [
        _style_question(
            question_id="bio_1",
            category=Category.BIOLOGY,
            subcategory="cell biology",
            question_type=QuestionType.TOSSUP,
            answer_mode=AnswerMode.SHORT_ANSWER,
            difficulty=2,
            question_text="What organelle generates ATP in this Science Bowl question?",
            answer_text="ANSWER: mitochondrion",
        )
    ]


def test_verifier_passes_grounded_short_answer_question() -> None:
    spec = _spec(spec_id="pass_short")
    bundle = _bundle(
        spec,
        ("cyclohexane", "textbook", "ch 1", "Cyclohexane is a cyclic saturated hydrocarbon containing six carbon atoms in a ring."),
    )
    draft = _draft(
        spec,
        question_text="What cyclic saturated hydrocarbon contains six carbon atoms in a ring in this Science Bowl question?",
        answer_text="ANSWER: cyclohexane",
        cited_chunk_ids=["cyclohexane"],
    )

    report = VerifierService().verify(spec, draft, bundle, _chemistry_styles())

    assert report.verdict == Verdict.PASS


def test_verifier_passes_grounded_multiple_choice_bonus() -> None:
    spec = _spec(
        spec_id="pass_mc",
        category=Category.CHEMISTRY,
        subcategory="general chemistry",
        question_type=QuestionType.BONUS,
        answer_mode=AnswerMode.MULTIPLE_CHOICE,
        difficulty=3,
    )
    bundle = _bundle(
        spec,
        ("neon", "textbook", "ch 2", "Neon is a noble gas that remains monatomic under standard conditions."),
    )
    draft = _draft(
        spec,
        question_text="In this Science Bowl question, which gas is noble at room temperature? W) oxygen X) neon Y) chlorine Z) nitrogen",
        answer_text="ANSWER: X) neon",
        cited_chunk_ids=["neon"],
        choices=[
            Choice(label="W", text="oxygen"),
            Choice(label="X", text="neon"),
            Choice(label="Y", text="chlorine"),
            Choice(label="Z", text="nitrogen"),
        ],
    )

    report = VerifierService().verify(spec, draft, bundle, _chemistry_styles())

    assert report.verdict == Verdict.PASS


def test_verifier_fails_missing_answer_prefix() -> None:
    spec = _spec(spec_id="missing_prefix")
    bundle = _bundle(
        spec,
        ("cyclohexane", "textbook", "ch 1", "Cyclohexane is a cyclic saturated hydrocarbon containing six carbon atoms in a ring."),
    )
    draft = _draft(
        spec,
        question_text="What cyclic saturated hydrocarbon contains six carbon atoms in a ring in this Science Bowl question?",
        answer_text="cyclohexane",
        cited_chunk_ids=["cyclohexane"],
    )

    report = VerifierService().verify(spec, draft, bundle, _chemistry_styles())

    assert report.verdict == Verdict.FAIL
    assert report.checks.format_compliance.issues[0].code == "missing_answer_prefix"


def test_verifier_fails_missing_citation() -> None:
    spec = _spec(spec_id="missing_citation")
    bundle = _bundle(
        spec,
        ("cyclohexane", "textbook", "ch 1", "Cyclohexane is a cyclic saturated hydrocarbon containing six carbon atoms in a ring."),
    )
    draft = _draft(
        spec,
        question_text="What cyclic saturated hydrocarbon contains six carbon atoms in a ring in this Science Bowl question?",
        answer_text="ANSWER: cyclohexane",
    )

    report = VerifierService().verify(spec, draft, bundle, _chemistry_styles())

    assert report.verdict == Verdict.FAIL
    assert report.checks.factual_grounding.issues[0].code == "missing_citation"


def test_verifier_fails_when_citation_exists_but_no_fact_chunks() -> None:
    spec = _spec(spec_id="missing_fact_chunk")
    bundle = RetrievalBundle(spec_id=spec.spec_id, fact_chunks=[])
    draft = _draft(
        spec,
        question_text="What cyclic saturated hydrocarbon contains six carbon atoms in a ring in this Science Bowl question?",
        answer_text="ANSWER: cyclohexane",
        cited_chunk_ids=["cyclohexane"],
    )

    report = VerifierService().verify(spec, draft, bundle, _chemistry_styles())

    assert report.verdict == Verdict.FAIL
    assert report.checks.factual_grounding.issues[0].code == "missing_fact_chunk"


def test_verifier_revises_when_question_is_not_in_science_bowl_format() -> None:
    spec = _spec(spec_id="style_generic")
    bundle = _bundle(
        spec,
        ("cyclohexane", "textbook", "ch 1", "Cyclohexane is a cyclic saturated hydrocarbon containing six carbon atoms in a ring."),
    )
    draft = _draft(
        spec,
        question_text="What cyclic saturated hydrocarbon contains six carbon atoms in a ring?",
        answer_text="ANSWER: cyclohexane",
        cited_chunk_ids=["cyclohexane"],
    )

    report = VerifierService().verify(spec, draft, bundle, _chemistry_styles())

    assert report.verdict == Verdict.REVISE
    assert report.checks.answerability.issues[0].code == "style_generic"


def test_verifier_revises_when_answer_text_is_effectively_empty() -> None:
    spec = _spec(spec_id="empty_answer")
    bundle = _bundle(
        spec,
        ("mitochondrion", "textbook", "ch 4", "The mitochondrion is the organelle responsible for most ATP production in eukaryotic cells."),
    )
    biology_spec = _spec(
        spec_id=spec.spec_id,
        category=Category.BIOLOGY,
        subcategory="cell biology",
        difficulty=2,
    )
    draft = _draft(
        biology_spec,
        question_text="What organelle generates ATP in this Science Bowl question?",
        answer_text="ANSWER: A",
        cited_chunk_ids=["mitochondrion"],
    )
    bundle = _bundle(
        biology_spec,
        ("mitochondrion", "textbook", "ch 4", "The mitochondrion is the organelle responsible for most ATP production in eukaryotic cells."),
    )

    report = VerifierService().verify(biology_spec, draft, bundle, _biology_styles())

    assert report.verdict == Verdict.REVISE
    assert report.checks.answerability.issues[0].code == "empty_answer"


def test_verifier_revises_on_large_difficulty_mismatch() -> None:
    spec = _spec(spec_id="difficulty_mismatch", difficulty=6)
    bundle = _bundle(
        spec,
        ("methane", "textbook", "ch 1", "Methane is the simplest alkane and contains one carbon atom."),
    )
    draft = _draft(
        spec,
        question_text="What is the simplest alkane in this Science Bowl question?",
        answer_text="ANSWER: methane",
        cited_chunk_ids=["methane"],
    )

    report = VerifierService().verify(spec, draft, bundle, _chemistry_styles())

    assert report.verdict == Verdict.REVISE
    assert report.checks.difficulty_alignment.issues[0].code == "difficulty_mismatch"


def test_style_alignment_can_fail_without_blocking_pass_verdict() -> None:
    spec = _spec(spec_id="style_missing", category=Category.ENERGY, subcategory="power systems")
    bundle = _bundle(
        spec,
        ("joule", "textbook", "ch 7", "A joule is the SI unit of energy."),
    )
    draft = _draft(
        spec,
        question_text="What SI unit measures energy in this Science Bowl question?",
        answer_text="ANSWER: joule",
        cited_chunk_ids=["joule"],
    )

    report = VerifierService().verify(spec, draft, bundle, _chemistry_styles())

    assert report.verdict == Verdict.PASS
    assert not report.checks.style_alignment.passed
    assert report.checks.style_alignment.issues[0].code == "missing_style_refs"


def test_novelty_can_fail_without_blocking_pass_verdict() -> None:
    spec = _spec(spec_id="novelty_fail")
    bundle = _bundle(
        spec,
        ("cyclohexane", "textbook", "ch 1", "Cyclohexane is a cyclic saturated hydrocarbon containing six carbon atoms in a ring."),
    )
    draft = _draft(
        spec,
        question_text="What cyclic saturated hydrocarbon contains six carbon atoms in a ring in this Science Bowl question?",
        answer_text="ANSWER: cyclohexane",
        cited_chunk_ids=["cyclohexane"],
    )

    report = VerifierService().verify(spec, draft, bundle, _chemistry_styles())

    assert report.verdict == Verdict.PASS
    assert not report.checks.novelty.passed
    assert report.checks.novelty.issues[0].code == "too_similar"


def test_bait_question_is_flagged_for_revision() -> None:
    spec = _spec(spec_id="bait_revise")
    bundle = _bundle(
        spec,
        ("cyclohexane", "textbook", "ch 1", "Cyclohexane is a cyclic saturated hydrocarbon containing six carbon atoms in a ring."),
        ("pentane", "textbook", "ch 1", "Pentane is a straight-chain alkane containing five carbon atoms."),
    )
    draft = _draft(
        spec,
        question_text="What straight-chain alkane contains five carbon atoms, but what cyclic saturated hydrocarbon contains six carbon atoms in a ring in this Science Bowl question?",
        answer_text="ANSWER: cyclohexane",
        cited_chunk_ids=["cyclohexane"],
    )

    report = VerifierService().verify(spec, draft, bundle, _chemistry_styles())

    assert report.verdict == Verdict.REVISE
    assert report.checks.bait_detection.issues[0].code == "bait_detected"


def test_bait_check_records_top_k_sampling_trace() -> None:
    spec = _spec(spec_id="bait_trace")
    bundle = _bundle(
        spec,
        ("cyclohexane", "textbook", "ch 1", "Cyclohexane is a cyclic saturated hydrocarbon containing six carbon atoms in a ring."),
        ("pentane", "textbook", "ch 1", "Pentane is a straight-chain alkane containing five carbon atoms."),
    )
    draft = _draft(
        spec,
        question_text="What straight-chain alkane contains five carbon atoms, but what cyclic saturated hydrocarbon contains six carbon atoms in a ring in this Science Bowl question?",
        answer_text="ANSWER: cyclohexane",
        cited_chunk_ids=["cyclohexane"],
    )

    checks = build_checks(spec, draft, bundle, _chemistry_styles())

    assert checks.bait_detection.sampled_prefixes
    assert all("top_k" in sample for sample in checks.bait_detection.sampled_prefixes)


def test_non_bait_question_passes_bait_detection() -> None:
    spec = _spec(spec_id="bait_pass")
    bundle = _bundle(
        spec,
        ("cyclohexane", "textbook", "ch 1", "Cyclohexane is a cyclic saturated hydrocarbon containing six carbon atoms in a ring."),
    )
    draft = _draft(
        spec,
        question_text="What cyclic saturated hydrocarbon contains six carbon atoms in a ring in this Science Bowl question?",
        answer_text="ANSWER: cyclohexane",
        cited_chunk_ids=["cyclohexane"],
    )

    report = VerifierService().verify(spec, draft, bundle, _chemistry_styles())

    assert report.checks.bait_detection.passed


def test_required_revisions_collect_bait_message() -> None:
    spec = _spec(spec_id="bait_revision_message")
    bundle = _bundle(
        spec,
        ("cyclohexane", "textbook", "ch 1", "Cyclohexane is a cyclic saturated hydrocarbon containing six carbon atoms in a ring."),
        ("pentane", "textbook", "ch 1", "Pentane is a straight-chain alkane containing five carbon atoms."),
    )
    draft = _draft(
        spec,
        question_text="What straight-chain alkane contains five carbon atoms, but what cyclic saturated hydrocarbon contains six carbon atoms in a ring in this Science Bowl question?",
        answer_text="ANSWER: cyclohexane",
        cited_chunk_ids=["cyclohexane"],
    )

    report = VerifierService().verify(spec, draft, bundle, _chemistry_styles())

    assert any("Top-k answer-prefix sampling found a strong early guess" in item for item in report.required_revisions)


def test_report_summary_reflects_pass_verdict() -> None:
    spec = _spec(spec_id="summary_pass")
    bundle = _bundle(
        spec,
        ("cyclohexane", "textbook", "ch 1", "Cyclohexane is a cyclic saturated hydrocarbon containing six carbon atoms in a ring."),
    )
    draft = _draft(
        spec,
        question_text="What cyclic saturated hydrocarbon contains six carbon atoms in a ring in this Science Bowl question?",
        answer_text="ANSWER: cyclohexane",
        cited_chunk_ids=["cyclohexane"],
    )

    report = VerifierService().verify(spec, draft, bundle, _chemistry_styles())

    assert report.summary == "Verification completed with verdict 'pass'."


def test_report_summary_reflects_revise_verdict() -> None:
    spec = _spec(spec_id="summary_revise")
    bundle = _bundle(
        spec,
        ("methane", "textbook", "ch 1", "Methane is the simplest alkane and contains one carbon atom."),
    )
    draft = _draft(
        spec,
        question_text="What is the simplest alkane?",
        answer_text="ANSWER: methane",
        cited_chunk_ids=["methane"],
    )

    report = VerifierService().verify(spec, draft, bundle, _chemistry_styles())

    assert report.summary == "Verification completed with verdict 'revise'."


def test_verifier_fail_overrides_other_checks_when_format_and_factual_both_break() -> None:
    spec = _spec(spec_id="double_fail")
    bundle = RetrievalBundle(spec_id=spec.spec_id, fact_chunks=[])
    draft = _draft(
        spec,
        question_text="What cyclic saturated hydrocarbon contains six carbon atoms in a ring?",
        answer_text="cyclohexane",
    )

    report = VerifierService().verify(spec, draft, bundle, _chemistry_styles())

    assert report.verdict == Verdict.FAIL
    assert not report.checks.format_compliance.passed
    assert not report.checks.factual_grounding.passed


def test_verifier_preserves_estimated_difficulty_for_short_clue() -> None:
    spec = _spec(spec_id="difficulty_estimate", difficulty=2)
    bundle = _bundle(
        spec,
        ("neon", "textbook", "ch 2", "Neon is a noble gas that remains monatomic under standard conditions."),
    )
    draft = _draft(
        spec,
        question_text="Which noble gas is neon in this Science Bowl question?",
        answer_text="ANSWER: neon",
        cited_chunk_ids=["neon"],
    )

    report = VerifierService().verify(spec, draft, bundle, _chemistry_styles())

    assert report.checks.difficulty_alignment.estimated_difficulty == 1


def test_verifier_handles_biology_question_correctly() -> None:
    spec = _spec(
        spec_id="bio_pass",
        category=Category.BIOLOGY,
        subcategory="cell biology",
        difficulty=2,
    )
    bundle = _bundle(
        spec,
        ("mitochondrion", "textbook", "ch 4", "The mitochondrion is the organelle responsible for most ATP production in eukaryotic cells."),
    )
    draft = _draft(
        spec,
        question_text="What organelle generates ATP in this Science Bowl question?",
        answer_text="ANSWER: mitochondrion",
        cited_chunk_ids=["mitochondrion"],
    )

    report = VerifierService().verify(spec, draft, bundle, _biology_styles())

    assert report.verdict == Verdict.PASS


def test_verifier_handles_physics_question_correctly() -> None:
    spec = _spec(
        spec_id="physics_pass",
        category=Category.PHYSICS,
        subcategory="mechanics",
        difficulty=2,
    )
    bundle = _bundle(
        spec,
        ("newton", "textbook", "ch 3", "The newton is the SI derived unit of force."),
    )
    styles = [
        _style_question(
            question_id="phys_1",
            category=Category.PHYSICS,
            subcategory="mechanics",
            question_type=QuestionType.TOSSUP,
            answer_mode=AnswerMode.SHORT_ANSWER,
            difficulty=2,
            question_text="What SI unit measures force in this Science Bowl question?",
            answer_text="ANSWER: newton",
        )
    ]
    draft = _draft(
        spec,
        question_text="What SI unit measures force in this Science Bowl question?",
        answer_text="ANSWER: newton",
        cited_chunk_ids=["newton"],
    )

    report = VerifierService().verify(spec, draft, bundle, styles)

    assert report.verdict == Verdict.PASS


def test_verifier_handles_earth_space_question_correctly() -> None:
    spec = _spec(
        spec_id="earth_pass",
        category=Category.EARTH_SPACE,
        subcategory="astronomy",
        difficulty=2,
    )
    bundle = _bundle(
        spec,
        ("mars", "textbook", "ch 8", "Mars is often called the Red Planet because of iron oxide on its surface."),
    )
    styles = [
        _style_question(
            question_id="earth_1",
            category=Category.EARTH_SPACE,
            subcategory="astronomy",
            question_type=QuestionType.TOSSUP,
            answer_mode=AnswerMode.SHORT_ANSWER,
            difficulty=2,
            question_text="Which planet is known as the Red Planet in this Science Bowl question?",
            answer_text="ANSWER: Mars",
        )
    ]
    draft = _draft(
        spec,
        question_text="Which planet is known as the Red Planet in this Science Bowl question?",
        answer_text="ANSWER: Mars",
        cited_chunk_ids=["mars"],
    )

    report = VerifierService().verify(spec, draft, bundle, styles)

    assert report.verdict == Verdict.PASS


def test_verifier_handles_math_question_correctly() -> None:
    spec = _spec(
        spec_id="math_pass",
        category=Category.MATH,
        subcategory="geometry",
        difficulty=2,
    )
    bundle = _bundle(
        spec,
        ("pi", "textbook", "ch 5", "Pi is the ratio of a circle's circumference to its diameter."),
    )
    styles = [
        _style_question(
            question_id="math_1",
            category=Category.MATH,
            subcategory="geometry",
            question_type=QuestionType.TOSSUP,
            answer_mode=AnswerMode.SHORT_ANSWER,
            difficulty=2,
            question_text="What constant is the ratio of a circle's circumference to its diameter in this Science Bowl question?",
            answer_text="ANSWER: pi",
        )
    ]
    draft = _draft(
        spec,
        question_text="What constant is the ratio of a circle's circumference to its diameter in this Science Bowl question?",
        answer_text="ANSWER: pi",
        cited_chunk_ids=["pi"],
    )

    report = VerifierService().verify(spec, draft, bundle, styles)

    assert report.verdict == Verdict.PASS


def test_verifier_handles_energy_question_correctly() -> None:
    spec = _spec(
        spec_id="energy_pass",
        category=Category.ENERGY,
        subcategory="electricity",
        difficulty=2,
    )
    bundle = _bundle(
        spec,
        ("watt", "textbook", "ch 9", "The watt is the SI derived unit of power."),
    )
    styles = [
        _style_question(
            question_id="energy_1",
            category=Category.ENERGY,
            subcategory="electricity",
            question_type=QuestionType.TOSSUP,
            answer_mode=AnswerMode.SHORT_ANSWER,
            difficulty=2,
            question_text="What SI unit measures power in this Science Bowl question?",
            answer_text="ANSWER: watt",
        )
    ]
    draft = _draft(
        spec,
        question_text="What SI unit measures power in this Science Bowl question?",
        answer_text="ANSWER: watt",
        cited_chunk_ids=["watt"],
    )

    report = VerifierService().verify(spec, draft, bundle, styles)

    assert report.verdict == Verdict.PASS


def test_bait_detection_does_not_trigger_on_simple_bonus_question() -> None:
    spec = _spec(
        spec_id="bonus_no_bait",
        category=Category.CHEMISTRY,
        subcategory="general chemistry",
        question_type=QuestionType.BONUS,
        answer_mode=AnswerMode.MULTIPLE_CHOICE,
        difficulty=3,
    )
    bundle = _bundle(
        spec,
        ("neon", "textbook", "ch 2", "Neon is a noble gas that remains monatomic under standard conditions."),
    )
    draft = _draft(
        spec,
        question_text="In this Science Bowl question, which gas is noble at room temperature? W) oxygen X) neon Y) chlorine Z) nitrogen",
        answer_text="ANSWER: X) neon",
        cited_chunk_ids=["neon"],
        choices=[
            Choice(label="W", text="oxygen"),
            Choice(label="X", text="neon"),
            Choice(label="Y", text="chlorine"),
            Choice(label="Z", text="nitrogen"),
        ],
    )

    report = VerifierService().verify(spec, draft, bundle, _chemistry_styles())

    assert report.checks.bait_detection.passed
