from __future__ import annotations

from pathlib import Path

import yaml

from scibowl.ingest.textbooks import ingest_textbook_text
from scibowl.schema.textbook import TextbookChunk
from scibowl.utils.ids import make_id
from scibowl.utils.io import read_jsonl, write_json, write_jsonl


def ingest_textbook_corpus(
    manifest_path: Path,
    raw_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    payload = yaml.safe_load(manifest_path.read_text(encoding='utf-8')) or {}
    sources = payload.get('sources')
    if not isinstance(sources, list) or not sources:
        raise ValueError('textbook manifest must contain a nonempty sources list')

    corpus_version = str(payload.get('corpus_version') or 'textbook_v2')
    max_words = int(payload.get('max_words') or 180)
    overlap_words = int(payload.get('overlap_words') or 30)
    source_summaries: list[dict[str, object]] = []
    total_chunks = 0
    corpus_run_id = make_id('corpus')

    for raw_source in sources:
        if not isinstance(raw_source, dict):
            raise ValueError('each textbook source must be a mapping')
        document_id = str(raw_source['document_id'])
        input_path = raw_dir / str(raw_source['file'])
        output_path = output_dir / f'{document_id}.jsonl'
        chunks = ingest_textbook_text(
            input_path,
            document_id=document_id,
            title=str(raw_source.get('title') or document_id),
            topics=[str(value) for value in raw_source.get('topics', [])],
            start_page=int(raw_source['start_page']),
            end_page=int(raw_source['end_page']),
            max_words=max_words,
            overlap_words=overlap_words,
            corpus_version=corpus_version,
            expected_sha256=str(raw_source['sha256']),
            expected_page_count=int(raw_source['page_count']),
            ingest_run_id=corpus_run_id,
        )
        write_jsonl(output_path, chunks)
        total_chunks += len(chunks)
        source_summaries.append(
            {
                'document_id': document_id,
                'input_path': str(input_path),
                'output_path': str(output_path),
                'chunk_count': len(chunks),
                'content_start_page': int(raw_source['start_page']),
                'content_end_page': int(raw_source['end_page']),
                'page_count': int(raw_source['page_count']),
                'sha256': str(raw_source['sha256']),
            }
        )

    summary = {
        'corpus_version': corpus_version,
        'corpus_run_id': corpus_run_id,
        'manifest_path': str(manifest_path),
        'raw_dir': str(raw_dir),
        'output_dir': str(output_dir),
        'source_count': len(source_summaries),
        'total_chunks': total_chunks,
        'max_words': max_words,
        'overlap_words': overlap_words,
        'sources': source_summaries,
    }
    write_json(output_dir / 'textbook_corpus_manifest.json', summary)
    return summary


def load_textbook_chunks(path: Path) -> list[TextbookChunk]:
    if path.is_dir():
        chunks: list[TextbookChunk] = []
        for jsonl_path in sorted(path.glob('*.jsonl')):
            chunks.extend(read_jsonl(jsonl_path, TextbookChunk))
    else:
        chunks = read_jsonl(path, TextbookChunk)

    chunk_ids: set[str] = set()
    versions: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in chunk_ids:
            raise ValueError(f'duplicate textbook chunk ID: {chunk.chunk_id}')
        chunk_ids.add(chunk.chunk_id)
        version = chunk.metadata.get('corpus_version')
        versions.add(str(version) if version is not None else 'legacy')
    if len(versions) > 1:
        raise ValueError(f'mixed textbook corpus versions are not allowed: {sorted(versions)}')
    return chunks
