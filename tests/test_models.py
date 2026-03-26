from scibowl.generate.local_model import (
    HeuristicWriterModel,
    PromptWriterModel,
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
