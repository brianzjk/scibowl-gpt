import json
import shutil
from pathlib import Path

from scibowl.eval.baseline import _limit_questions_per_category
from scibowl.eval.splits import build_rated_question_split
from scibowl.schema.common import AnswerMode, Category, QuestionType, SourceType
from scibowl.schema.question import NormalizedQuestion
from scibowl.utils.ids import make_id


def test_build_rated_question_split() -> None:
    tmp_path = Path("tests_runtime") / make_id("split")
    tmp_path.mkdir(parents=True, exist_ok=True)
    questions_path = tmp_path / "questions.jsonl"
    reviews_path = tmp_path / "reviews.jsonl"

    questions_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "question_id": f"q{i}",
                        "source_type": "dataset",
                        "source_id": "set",
                        "category": "biology",
                        "subcategory": "cell biology",
                        "question_type": "tossup",
                        "answer_mode": "short_answer",
                        "difficulty": 3,
                        "question_text": f"Question {i}",
                        "answer_text": "ANSWER: test",
                    }
                )
                for i in range(10)
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    reviews_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "review_id": f"r{i}",
                        "question_id": f"q{i}",
                        "reviewer_id": "agg",
                        "ratings": {"quality": 0.5, "difficulty": 3},
                    }
                )
                for i in range(10)
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    split = build_rated_question_split(questions_path, reviews_path, seed=1)

    assert split["counts"]["train"] == 8
    assert split["counts"]["val"] == 1
    assert split["counts"]["test"] == 1
    shutil.rmtree(tmp_path)


def test_limit_questions_per_category() -> None:
    questions = [
        NormalizedQuestion(
            question_id=f"bio{i}",
            source_type=SourceType.DATASET,
            source_id="set",
            category=Category.BIOLOGY,
            subcategory="cells",
            question_type=QuestionType.TOSSUP,
            answer_mode=AnswerMode.SHORT_ANSWER,
            difficulty=3,
            question_text=f"Biology {i}",
            answer_text="ANSWER: cell",
        )
        for i in range(3)
    ] + [
        NormalizedQuestion(
            question_id=f"chem{i}",
            source_type=SourceType.DATASET,
            source_id="set",
            category=Category.CHEMISTRY,
            subcategory="equilibrium",
            question_type=QuestionType.TOSSUP,
            answer_mode=AnswerMode.SHORT_ANSWER,
            difficulty=3,
            question_text=f"Chemistry {i}",
            answer_text="ANSWER: equilibrium",
        )
        for i in range(3)
    ]

    limited = _limit_questions_per_category(questions, 2)

    assert [question.question_id for question in limited] == ["bio0", "bio1", "chem0", "chem1"]
