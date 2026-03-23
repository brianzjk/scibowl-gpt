import json
import shutil
from pathlib import Path

from scibowl.eval.splits import build_rated_question_split
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
