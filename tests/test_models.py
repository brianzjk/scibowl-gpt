from scibowl.generate.local_model import (
    HeuristicWriterModel,
    PromptWriterModel,
    _extract_writer_choices,
    build_generated_draft,
    build_writer_model,
)
from scibowl.schema.common import AnswerMode, Category, QuestionType
from scibowl.schema.generation import QuestionSpec, RetrievalBundle


def test_build_writer_model_defaults_to_heuristic(monkeypatch) -> None:
    monkeypatch.delenv("SCIBOWL_WRITER_PROVIDER", raising=False)
    monkeypatch.delenv("SCIBOWL_WRITER_MODEL", raising=False)
    monkeypatch.delenv("SCIBOWL_WRITER_BASE_URL", raising=False)

    model = build_writer_model()

    assert isinstance(model, HeuristicWriterModel)


def test_prompt_writer_model_falls_back_on_invalid_json() -> None:
    class BrokenClient:
        def complete_json(self, **kwargs):
            raise ValueError("bad json")

    model = PromptWriterModel(client=BrokenClient(), model_name="broken-model")
    spec = QuestionSpec(
        spec_id="spec_1",
        category=Category.CHEMISTRY,
        subcategory="equilibrium",
        question_type=QuestionType.TOSSUP,
        difficulty=4,
        topic_focus=["Le Chatelier's principle"],
    )

    draft = model.generate(spec, RetrievalBundle(spec_id=spec.spec_id))

    assert draft.model_info.model_name == "broken-model-fallback"
    assert draft.question.answer_text.startswith("ANSWER:")


def test_prompt_writer_model_retries_on_invalid_payload(monkeypatch) -> None:
    class FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete_json(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"process": "thinking"}
            return {
                "question_text": "Which of the following is a sedimentary rock?",
                "answer_text": "ANSWER: X) limestone",
                "choices": [
                    {"W": "granite"},
                    {"X": "limestone"},
                    {"Y": "marble"},
                    {"Z": "basalt"},
                ],
            }

    monkeypatch.setenv("SCIBOWL_WRITER_RETRIES", "2")
    client = FlakyClient()
    model = PromptWriterModel(client=client, model_name="flaky-model")
    spec = QuestionSpec(
        spec_id="spec_retry",
        category=Category.EARTH_SPACE,
        subcategory="Rocks and Minerals",
        question_type=QuestionType.TOSSUP,
        difficulty=3,
        topic_focus=["sedimentary rocks"],
    )

    draft = model.generate(spec, RetrievalBundle(spec_id=spec.spec_id))

    assert client.calls == 2
    assert draft.model_info.model_name == "flaky-model"
    assert draft.question.answer_mode == AnswerMode.MULTIPLE_CHOICE


def test_generated_draft_infers_answer_mode_from_writer_output() -> None:
    spec = QuestionSpec(
        spec_id="spec_mc_infer",
        category=Category.CHEMISTRY,
        subcategory="equilibrium",
        question_type=QuestionType.BONUS,
        difficulty=4,
        topic_focus=["Le Chatelier's principle"],
    )

    draft = build_generated_draft(
        spec=spec,
        bundle=RetrievalBundle(spec_id=spec.spec_id),
        model_info=HeuristicWriterModel().model_info,
        question_text="Which of the following best describes the effect of adding reactant?",
        answer_text="ANSWER: X) shifts right",
        choices=[
            {"label": "W", "text": "shifts left"},
            {"label": "X", "text": "shifts right"},
            {"label": "Y", "text": "no change"},
            {"label": "Z", "text": "cannot be determined"},
        ],
    )

    assert draft.question.answer_mode == AnswerMode.MULTIPLE_CHOICE


def test_extract_writer_choices_accepts_common_choice_shapes() -> None:
    choices = _extract_writer_choices(
        {
            "choices": [
                {"W": "igneous rock"},
                "X) sedimentary rock",
                {"choice_label": "Y", "choice_text": "metamorphic rock"},
                {"label": "Z", "text": "magma"},
            ]
        }
    )

    assert [(choice.label, choice.text) for choice in choices] == [
        ("W", "igneous rock"),
        ("X", "sedimentary rock"),
        ("Y", "metamorphic rock"),
        ("Z", "magma"),
    ]
