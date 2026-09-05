from __future__ import annotations

import json
from pathlib import Path

from scibowl.schema.training import CleanSFTExample
from scibowl.utils.io import read_jsonl


TINKER_SPLITS = ('train', 'val', 'test', 'held_out')


def validate_tinker_sft_dataset(data_dir: Path) -> dict[str, object]:
    """Validate a clean export before it is sent to a paid training run."""
    counts: dict[str, int] = {}
    ids_by_split: dict[str, set[str]] = {}
    groups_by_split: dict[str, set[str]] = {}

    for split in TINKER_SPLITS:
        path = data_dir / f'clean_sft_{split}.jsonl'
        if not path.is_file():
            raise FileNotFoundError(f'Missing required split: {path}')
        examples = read_jsonl(path, CleanSFTExample)
        counts[split] = len(examples)
        ids_by_split[split] = set()
        groups_by_split[split] = set()
        for example in examples:
            if example.metadata.split != split:
                raise ValueError(
                    f'{example.example_id} says split={example.metadata.split!r} in {path.name}'
                )
            if example.example_id in ids_by_split[split]:
                raise ValueError(f'Duplicate example_id: {example.example_id}')
            ids_by_split[split].add(example.example_id)
            groups_by_split[split].add(example.metadata.duplicate_group)
            _validate_messages(example)

    for index, split in enumerate(TINKER_SPLITS):
        for other in TINKER_SPLITS[index + 1 :]:
            if ids_by_split[split] & ids_by_split[other]:
                raise ValueError(f'Example IDs overlap between {split} and {other}')
            if groups_by_split[split] & groups_by_split[other]:
                raise ValueError(f'Duplicate groups overlap between {split} and {other}')

    if not counts['train']:
        raise ValueError('Training split is empty')
    if not counts['val']:
        raise ValueError('Validation split is empty')

    return {
        'schema_version': 'tinker_sft_validation_v1',
        'data_dir': str(data_dir),
        'split_counts': counts,
        'total_examples': sum(counts.values()),
    }


def _validate_messages(example: CleanSFTExample) -> None:
    roles = [message.role for message in example.messages]
    if roles != ['system', 'user', 'assistant']:
        raise ValueError(f'{example.example_id} has unsupported message roles: {roles}')
    if any(not message.content.strip() for message in example.messages):
        raise ValueError(f'{example.example_id} has an empty message')
    try:
        answer = json.loads(example.messages[-1].content)
    except json.JSONDecodeError as exc:
        raise ValueError(f'{example.example_id} has invalid assistant JSON') from exc
    if not isinstance(answer, dict) or set(answer) != {
        'question_text',
        'answer_text',
        'choices',
    }:
        raise ValueError(f'{example.example_id} has an invalid assistant response shape')
    if not isinstance(answer['question_text'], str) or not answer['question_text'].strip():
        raise ValueError(f'{example.example_id} has an empty question_text')
    if not isinstance(answer['answer_text'], str) or not answer['answer_text'].startswith('ANSWER:'):
        raise ValueError(f'{example.example_id} has an invalid answer_text')
    if not isinstance(answer['choices'], list):
        raise ValueError(f'{example.example_id} has invalid choices')
