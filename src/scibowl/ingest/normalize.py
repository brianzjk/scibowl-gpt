from __future__ import annotations

from pathlib import Path

from scibowl.schema.question import NormalizedQuestion
from scibowl.utils.io import read_jsonl, write_jsonl


def validate_normalized_questions(path: Path) -> list[NormalizedQuestion]:
    return read_jsonl(path, NormalizedQuestion)


def rewrite_normalized_questions(input_path: Path, output_path: Path) -> int:
    rows = validate_normalized_questions(input_path)
    write_jsonl(output_path, rows)
    return len(rows)
