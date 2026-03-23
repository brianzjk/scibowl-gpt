from pathlib import Path

from scibowl.schema.common import AnswerMode, Category, QuestionType, SourceType
from scibowl.schema.generation import QuestionSpec
from scibowl.schema.manifest import DatasetManifest
from scibowl.schema.question import NormalizedQuestion
from scibowl.utils.io import read_json


def test_question_spec_validates() -> None:
    spec = QuestionSpec(
        spec_id="spec_1",
        category=Category.EARTH_SPACE,
        subcategory="stars",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
    )
    assert spec.difficulty == 4


def test_normalized_question_validates() -> None:
    question = NormalizedQuestion(
        question_id="q1",
        source_type=SourceType.DATASET,
        source_id="set1",
        category=Category.EARTH_SPACE,
        subcategory="stars",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=3,
        question_text="What is the closest star to Earth after the Sun?",
        answer_text="ANSWER: Proxima Centauri",
    )
    assert question.answer_text.startswith("ANSWER:")


def test_dataset_manifest_validates() -> None:
    manifest = DatasetManifest.model_validate(
        read_json(Path("data/processed/manifests/mit_2025_rated_corpus.json"))
    )
    assert manifest.role == "rated_corpus"
