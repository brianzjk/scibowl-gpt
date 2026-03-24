from scibowl.generate.local_model import HeuristicWriterModel, PromptWriterModel, build_writer_model
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
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
        topic_focus=["Le Chatelier's principle"],
    )

    draft = model.generate(spec, RetrievalBundle(spec_id=spec.spec_id))

    assert draft.model_info.model_name == "broken-model-fallback"
    assert draft.question.answer_text.startswith("ANSWER:")
