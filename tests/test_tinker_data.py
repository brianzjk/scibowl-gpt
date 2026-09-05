import json
import shutil
from pathlib import Path

import pytest

from scibowl.schema.common import AnswerMode, Category, QuestionType
from scibowl.schema.training import ChatMessage, CleanSFTExample, CleanSFTMetadata
from scibowl.training.tinker import TINKER_SPLITS, validate_tinker_sft_dataset
from scibowl.utils.ids import make_id
from scibowl.utils.io import write_jsonl


def _example(example_id: str, split: str, group: str) -> CleanSFTExample:
    return CleanSFTExample(
        example_id=example_id,
        messages=[
            ChatMessage(role='system', content='Write a question.'),
            ChatMessage(role='user', content='Category: Math'),
            ChatMessage(
                role='assistant',
                content=json.dumps(
                    {
                        'question_text': 'What is one plus one?',
                        'answer_text': 'ANSWER: two',
                        'choices': [],
                    }
                ),
            ),
        ],
        metadata=CleanSFTMetadata(
            source_question_id=example_id,
            source_dataset='mit_2026',
            source_family='mit',
            source_year=2026,
            category=Category.MATH,
            question_type=QuestionType.TOSSUP,
            target_answer_mode=AnswerMode.SHORT_ANSWER,
            profile='mit_recent',
            split=split,
            duplicate_group=group,
            exact_fingerprint=example_id,
            near_fingerprint=group,
        ),
    )


def test_validate_tinker_sft_dataset_checks_all_fixed_splits() -> None:
    data_dir = Path('tests_runtime') / make_id('tinker')
    for split in TINKER_SPLITS:
        write_jsonl(data_dir / f'clean_sft_{split}.jsonl', [_example(split, split, split)])

    summary = validate_tinker_sft_dataset(data_dir)

    assert summary['split_counts'] == {split: 1 for split in TINKER_SPLITS}
    shutil.rmtree(data_dir)


def test_validate_tinker_sft_dataset_rejects_cross_split_duplicates() -> None:
    data_dir = Path('tests_runtime') / make_id('tinker')
    for split in TINKER_SPLITS:
        group = 'shared' if split in {'train', 'val'} else split
        write_jsonl(data_dir / f'clean_sft_{split}.jsonl', [_example(split, split, group)])

    with pytest.raises(ValueError, match='Duplicate groups overlap'):
        validate_tinker_sft_dataset(data_dir)
    shutil.rmtree(data_dir)
