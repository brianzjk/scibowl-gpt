from __future__ import annotations

import random
from pathlib import Path

from scibowl.schema.question import NormalizedQuestion
from scibowl.schema.review import HumanReview
from scibowl.utils.io import read_jsonl


def build_rated_question_split(
    questions_path: Path,
    reviews_path: Path,
    *,
    seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict[str, object]:
    questions = read_jsonl(questions_path, NormalizedQuestion)
    reviews = read_jsonl(reviews_path, HumanReview)

    rated_question_ids = sorted({review.question_id for review in reviews})
    question_lookup = {question.question_id: question for question in questions}
    rated_question_ids = [question_id for question_id in rated_question_ids if question_id in question_lookup]

    rng = random.Random(seed)
    rng.shuffle(rated_question_ids)

    train_end = int(len(rated_question_ids) * train_fraction)
    val_end = train_end + int(len(rated_question_ids) * val_fraction)

    split = {
        "split_id": f"rated_split_seed_{seed}",
        "question_set_path": str(questions_path),
        "reviews_path": str(reviews_path),
        "seed": seed,
        "counts": {
            "train": train_end,
            "val": val_end - train_end,
            "test": len(rated_question_ids) - val_end,
        },
        "train_question_ids": rated_question_ids[:train_end],
        "val_question_ids": rated_question_ids[train_end:val_end],
        "test_question_ids": rated_question_ids[val_end:],
    }
    return split
