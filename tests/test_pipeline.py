from scibowl.generate.orchestration import GenerationOrchestrator
from scibowl.retrieval.search import retrieve_bundle
from scibowl.schema.common import AnswerMode, Category, Citation, ModelInfo, QuestionType, SourceType, Verdict
from scibowl.schema.generation import DraftQuestion, GeneratedDraft, QuestionSpec
from scibowl.schema.question import NormalizedQuestion
from scibowl.schema.textbook import TextbookChunk
from scibowl.verify.pipeline import VerifierService
from scibowl.verify.rules import build_checks


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
            document_id="seeds_foundations_of_astrophysics",
            title="Foundations of Astrophysics",
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


def test_retrieve_bundle_filters_textbooks_by_subject_manifest() -> None:
    spec = QuestionSpec(
        spec_id="spec_2",
        category=Category.BIOLOGY,
        subcategory="cells",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
        topic_focus=["membrane transport"],
    )
    chunks = [
        TextbookChunk(
            chunk_id="bio1",
            document_id="campbell_biology_12e",
            title="Campbell Biology 12th Edition",
            topics=["cells"],
            text="Cell membranes regulate transport through channels and pumps.",
            char_count=61,
            token_count_est=9,
            metadata={},
        ),
        TextbookChunk(
            chunk_id="chem1",
            document_id="atkins_general_chemistry",
            title="Atkins General Chemistry",
            topics=["equilibrium"],
            text="Chemical equilibrium shifts in response to concentration changes.",
            char_count=68,
            token_count_est=9,
            metadata={},
        ),
    ]

    bundle = retrieve_bundle(spec, chunks, [])

    assert [chunk.document_id for chunk in bundle.fact_chunks] == ["campbell_biology_12e"]


def test_retrieve_bundle_routes_earth_space_books_by_subcategory() -> None:
    chunks = [
        TextbookChunk(
            chunk_id="seed1",
            document_id="seeds_foundations_of_astrophysics",
            title="Foundations of Astrophysics",
            topics=["stars"],
            text="Stars produce light by nuclear fusion.",
            char_count=38,
            token_count_est=6,
            metadata={},
        ),
        TextbookChunk(
            chunk_id="tar1",
            document_id="tarbuck_earth_science",
            title="Earth Science",
            topics=["hydrology"],
            text="Groundwater moves through permeable rock layers.",
            char_count=48,
            token_count_est=7,
            metadata={},
        ),
        TextbookChunk(
            chunk_id="met1",
            document_id="ahrens_essentials_of_meteorology",
            title="Essentials of Meteorology",
            topics=["meteorology"],
            text="Cold fronts, pressure gradients, humidity, and clouds drive weather patterns in the atmosphere.",
            char_count=90,
            token_count_est=14,
            metadata={},
        ),
        TextbookChunk(
            chunk_id="ocn1",
            document_id="garrison_essentials_of_oceanography_5e",
            title="Essentials of Oceanography",
            topics=["hydrology", "oceanography"],
            text="Runoff, watersheds, infiltration, and groundwater connect the ocean to the broader water cycle.",
            char_count=94,
            token_count_est=14,
            metadata={},
        ),
        TextbookChunk(
            chunk_id="obs1",
            document_id="burns_practical_observational_astronomy",
            title="A Practical Guide to Observational Astronomy",
            topics=["observation", "astronomy"],
            text="Horizon coordinates, equatorial coordinates, telescope mounts, detectors, and photometry are central to observational astronomy.",
            char_count=125,
            token_count_est=18,
            metadata={},
        ),
    ]

    astro_spec = QuestionSpec(
        spec_id="spec_ast",
        category=Category.EARTH_SPACE,
        subcategory="Stars",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
        topic_focus=["stellar fusion"],
    )
    earth_spec = QuestionSpec(
        spec_id="spec_earth",
        category=Category.EARTH_SPACE,
        subcategory="Hydrology",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
        topic_focus=["groundwater"],
    )
    meteo_spec = QuestionSpec(
        spec_id="spec_meteo",
        category=Category.EARTH_SPACE,
        subcategory="Meteorology",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
        topic_focus=["cold fronts"],
    )
    observation_spec = QuestionSpec(
        spec_id="spec_observation",
        category=Category.EARTH_SPACE,
        subcategory="Observation",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
        topic_focus=["equatorial coordinates"],
    )
    tectonics_spec = QuestionSpec(
        spec_id="spec_tectonics",
        category=Category.EARTH_SPACE,
        subcategory="Tectonics",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
        topic_focus=["plate boundaries"],
    )

    astro_bundle = retrieve_bundle(astro_spec, chunks, [])
    earth_bundle = retrieve_bundle(earth_spec, chunks, [])
    meteo_bundle = retrieve_bundle(meteo_spec, chunks, [])
    observation_bundle = retrieve_bundle(observation_spec, chunks, [])
    tectonics_bundle = retrieve_bundle(tectonics_spec, chunks, [])

    assert [chunk.document_id for chunk in astro_bundle.fact_chunks] == ["seeds_foundations_of_astrophysics"]
    assert [chunk.document_id for chunk in earth_bundle.fact_chunks][:1] == ["garrison_essentials_of_oceanography_5e"]
    assert "tarbuck_earth_science" in [chunk.document_id for chunk in earth_bundle.fact_chunks]
    assert [chunk.document_id for chunk in meteo_bundle.fact_chunks] == ["ahrens_essentials_of_meteorology"]
    assert [chunk.document_id for chunk in observation_bundle.fact_chunks] == ["burns_practical_observational_astronomy"]
    assert [chunk.document_id for chunk in tectonics_bundle.fact_chunks] == ["tarbuck_earth_science"]


def test_build_checks_flags_textbook_meta_questions_and_bad_multiple_choice() -> None:
    spec = QuestionSpec(
        spec_id="spec_3",
        category=Category.CHEMISTRY,
        subcategory="equilibrium",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.MULTIPLE_CHOICE,
        difficulty=4,
        topic_focus=["Le Chatelier's principle"],
    )
    draft = GeneratedDraft(
        draft_id="draft_1",
        spec_id=spec.spec_id,
        model_info=ModelInfo(provider="local", model_name="test", prompt_version="writer_v1"),
        question=DraftQuestion(
            category=Category.CHEMISTRY,
            subcategory="equilibrium",
            question_type=QuestionType.TOSSUP,
            answer_mode=AnswerMode.MULTIPLE_CHOICE,
            difficulty=4,
            question_text="According to the textbook, which chapter covers equilibrium? W) 1 X) 2 Y) 3 Z) 4",
            answer_text="ANSWER: X) 2",
            choices=[],
        ),
    )

    checks = build_checks(spec, draft, retrieve_bundle(spec, [], []), [])
    answerability_messages = [issue.message for issue in checks.answerability.issues]
    format_messages = [issue.message for issue in checks.format_compliance.issues]

    assert any("source material" in message for message in answerability_messages)
    assert any("four choices labeled W, X, Y, Z" in message for message in format_messages)
    assert any("not embedded in question_text" in message for message in format_messages)


def test_build_checks_flags_sentence_length_short_answer_lines() -> None:
    spec = QuestionSpec(
        spec_id="spec_sentence_answer",
        category=Category.EARTH_SPACE,
        subcategory="Hydrology",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
        topic_focus=["groundwater"],
    )
    draft = GeneratedDraft(
        draft_id="draft_sentence_answer",
        spec_id=spec.spec_id,
        model_info=ModelInfo(provider="local", model_name="test", prompt_version="writer_v1"),
        question=DraftQuestion(
            category=Category.EARTH_SPACE,
            subcategory="Hydrology",
            question_type=QuestionType.TOSSUP,
            answer_mode=AnswerMode.SHORT_ANSWER,
            difficulty=4,
            question_text="What is the name of the process by which water enters soil from the surface?",
            answer_text="ANSWER: It is the process in which water moves from the surface into the soil.",
        ),
    )

    checks = build_checks(spec, draft, retrieve_bundle(spec, [], []), [])

    assert any(issue.code == "sentence_answer_for_short_answer" for issue in checks.answerability.issues)


def test_build_checks_flags_multiple_choice_without_required_phrase() -> None:
    spec = QuestionSpec(
        spec_id="spec_missing_phrase",
        category=Category.CHEMISTRY,
        subcategory="equilibrium",
        question_type=QuestionType.BONUS,
        answer_mode=AnswerMode.MULTIPLE_CHOICE,
        difficulty=4,
        topic_focus=["equilibrium"],
    )
    draft = GeneratedDraft(
        draft_id="draft_missing_phrase",
        spec_id=spec.spec_id,
        model_info=ModelInfo(provider="local", model_name="test", prompt_version="writer_v1"),
        question=DraftQuestion(
            category=Category.CHEMISTRY,
            subcategory="equilibrium",
            question_type=QuestionType.BONUS,
            answer_mode=AnswerMode.MULTIPLE_CHOICE,
            difficulty=4,
            question_text="At equilibrium, adding more reactant shifts the system in which direction?",
            answer_text="ANSWER: to the right",
            choices=[
                {"label": "W", "text": "to the left"},
                {"label": "X", "text": "to the right"},
                {"label": "Y", "text": "no change"},
                {"label": "Z", "text": "cannot be determined"},
            ],
        ),
    )

    checks = build_checks(spec, draft, retrieve_bundle(spec, [], []), [])

    assert any(issue.code == "missing_which_of_the_following" for issue in checks.format_compliance.issues)


def test_build_checks_flags_supplemental_source_chunks() -> None:
    spec = QuestionSpec(
        spec_id="spec_4",
        category=Category.EARTH_SPACE,
        subcategory="stars",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
        topic_focus=["stellar spectra"],
    )
    draft = GeneratedDraft(
        draft_id="draft_2",
        spec_id=spec.spec_id,
        model_info=ModelInfo(provider="local", model_name="test", prompt_version="writer_v1"),
        question=DraftQuestion(
            category=Category.EARTH_SPACE,
            subcategory="stars",
            question_type=QuestionType.TOSSUP,
            answer_mode=AnswerMode.SHORT_ANSWER,
            difficulty=4,
            question_text="Which stars show the strongest Balmer lines?",
            answer_text="ANSWER: A-type stars",
        ),
        citations=[Citation(source_id="astro", chunk_id="chunk1", locator=None)],
    )
    bundle = retrieve_bundle(
        spec,
        [
            TextbookChunk(
                chunk_id="chunk1",
                document_id="seeds_foundations_of_astrophysics",
                title="Foundations of Astrophysics",
                topics=["stars"],
                text="The companion website LaunchPad includes simulations about stellar spectra.",
                char_count=74,
                token_count_est=10,
                metadata={},
            )
        ],
        [],
    )

    checks = build_checks(spec, draft, bundle, [])

    assert any(issue.code == "suspicious_source_chunk" for issue in checks.factual_grounding.issues)


def test_verifier_service_merges_solver_based_llm_feedback() -> None:
    spec = QuestionSpec(
        spec_id="spec_5",
        category=Category.EARTH_SPACE,
        subcategory="stars",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
        topic_focus=["Balmer lines"],
    )
    draft = GeneratedDraft(
        draft_id="draft_3",
        spec_id=spec.spec_id,
        model_info=ModelInfo(provider="local", model_name="test", prompt_version="writer_v1"),
        question=DraftQuestion(
            category=Category.EARTH_SPACE,
            subcategory="stars",
            question_type=QuestionType.TOSSUP,
            answer_mode=AnswerMode.SHORT_ANSWER,
            difficulty=4,
            question_text="Which stars show the strongest Balmer lines?",
            answer_text="ANSWER: O-type stars",
        ),
        citations=[Citation(source_id="astro", chunk_id="chunk1")],
    )
    bundle = retrieve_bundle(
        spec,
        [
            TextbookChunk(
                chunk_id="chunk1",
                document_id="seeds_foundations_of_astrophysics",
                title="Foundations of Astrophysics",
                topics=["stars"],
                text="A-type stars show the strongest Balmer absorption lines in their spectra.",
                char_count=75,
                token_count_est=11,
                metadata={},
            )
        ],
        [],
    )
    styles = [
        NormalizedQuestion(
            question_id="style_1",
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

    class FakeVerifierModel:
        model_name = "fake-verifier"
        prompt_version = "verifier_v1"

        def review(self, spec, draft, bundle):
            return {
                "summary": "Question is on topic, but the solved answer does not match the provided answer.",
                "format_issues": ["Tighten the question wording into a cleaner interrogative sentence."],
                "topic_issues": [],
                "style_issues": ["Question is too much of a reaction-speed test."],
                "scientific_accuracy_issues": ["Solving the question gives A-type stars, not O-type stars."],
                "required_revisions": ["Fix the answer so it matches the science."],
                "solver_answer": "A-type stars",
                "solver_answer_matches_expected": False,
                "topic_score": 0.9,
                "style_score": 0.6,
            }

    report = VerifierService(llm_model=FakeVerifierModel()).verify(spec, draft, bundle, styles)

    assert report.verdict == Verdict.FAIL
    assert any(issue.code == "llm_format_issue" for issue in report.checks.format_compliance.issues)
    assert any(issue.code == "llm_scientific_accuracy_issue" for issue in report.checks.factual_grounding.issues)
    assert any(issue.code == "solver_answer_mismatch" for issue in report.checks.factual_grounding.issues)
    assert any(issue.code == "llm_style_issue" for issue in report.checks.style_alignment.issues)
    assert report.checks.topic_alignment.style_score == 0.9
    assert report.checks.style_alignment.style_score == 0.6
    assert "Fix the answer so it matches the science." in report.required_revisions
