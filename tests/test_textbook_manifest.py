import hashlib
import shutil
from pathlib import Path

import yaml

from scibowl.ingest.textbook_corpus import ingest_textbook_corpus, load_textbook_chunks
from scibowl.utils.ids import make_id


def test_textbook_manifest_has_fixed_bounds_and_unique_sources() -> None:
    payload = yaml.safe_load(
        Path('configs/textbook_sources.yaml').read_text(encoding='utf-8')
    )
    sources = payload['sources']

    assert len(sources) == 11
    assert len({source['document_id'] for source in sources}) == 11
    assert len({source['file'] for source in sources}) == 11
    assert all(1 <= source['start_page'] <= source['end_page'] <= source['page_count'] for source in sources)
    assert all(len(source['sha256']) == 64 for source in sources)


def test_corpus_builder_uses_one_shared_run_id() -> None:
    tmp_path = Path('tests_runtime') / make_id('corpus_manifest')
    raw_dir = tmp_path / 'raw'
    output_dir = tmp_path / 'out'
    raw_dir.mkdir(parents=True)
    text_path = raw_dir / 'book.txt'
    text_path.write_text('A short scientific source about conserved energy.', encoding='utf-8')
    source_hash = hashlib.sha256(text_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / 'sources.yaml'
    manifest_path.write_text(
        yaml.safe_dump(
            {
                'corpus_version': 'test_v1',
                'max_words': 20,
                'overlap_words': 5,
                'sources': [
                    {
                        'document_id': 'book',
                        'file': 'book.txt',
                        'title': 'Book',
                        'topics': ['physics'],
                        'start_page': 1,
                        'end_page': 1,
                        'page_count': 1,
                        'sha256': source_hash,
                    }
                ],
            }
        ),
        encoding='utf-8',
    )

    summary = ingest_textbook_corpus(manifest_path, raw_dir, output_dir)
    chunks = load_textbook_chunks(output_dir)

    assert summary['corpus_run_id'].startswith('corpus_')
    assert {chunk.metadata['ingest_run_id'] for chunk in chunks} == {
        summary['corpus_run_id']
    }
    shutil.rmtree(tmp_path)
