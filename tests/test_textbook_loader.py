import shutil
from pathlib import Path

from scibowl.ingest.textbook_corpus import load_textbook_chunks
from scibowl.schema.textbook import TextbookChunk
from scibowl.utils.ids import make_id
from scibowl.utils.io import write_jsonl


def _chunk(chunk_id: str, version: str) -> TextbookChunk:
    return TextbookChunk(
        chunk_id=chunk_id,
        document_id='book',
        title='Book',
        text='Useful science text.',
        char_count=20,
        token_count_est=3,
        metadata={'corpus_version': version},
    )


def test_textbook_loader_rejects_mixed_versions_and_duplicate_ids() -> None:
    root = Path('tests_runtime') / make_id('loader')
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / 'a.jsonl', [_chunk('a', 'v1')])
    write_jsonl(root / 'b.jsonl', [_chunk('b', 'v2')])

    try:
        load_textbook_chunks(root)
    except ValueError as exc:
        assert 'mixed textbook corpus versions' in str(exc)
    else:
        raise AssertionError('mixed versions must raise ValueError')

    write_jsonl(root / 'b.jsonl', [_chunk('a', 'v1')])
    try:
        load_textbook_chunks(root)
    except ValueError as exc:
        assert 'duplicate textbook chunk ID' in str(exc)
    else:
        raise AssertionError('duplicate IDs must raise ValueError')
    shutil.rmtree(root)
