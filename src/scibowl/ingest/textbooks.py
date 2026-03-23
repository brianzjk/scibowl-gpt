from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from scibowl.schema.textbook import TextbookChunk
from scibowl.utils.ids import make_id, slugify
from scibowl.utils.text import chunk_text, estimate_token_count, normalize_whitespace


def _read_textbook_text(input_path: Path) -> str:
    if input_path.suffix.lower() == ".pdf":
        reader = PdfReader(str(input_path))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
    return input_path.read_text(encoding="utf-8")


def ingest_textbook_text(
    input_path: Path,
    document_id: str | None = None,
    title: str | None = None,
    topics: list[str] | None = None,
) -> list[TextbookChunk]:
    raw_text = _read_textbook_text(input_path)
    normalized = normalize_whitespace(raw_text)
    doc_id = document_id or slugify(input_path.stem)
    doc_title = title or input_path.stem
    topic_list = topics or []

    chunks: list[TextbookChunk] = []
    for index, chunk in enumerate(chunk_text(normalized), start=1):
        chunk_id = f"{doc_id}_{index:04d}"
        chunks.append(
            TextbookChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                title=doc_title,
                chapter=None,
                section=None,
                pages=[],
                topics=topic_list,
                text=chunk,
                char_count=len(chunk),
                token_count_est=estimate_token_count(chunk),
                metadata={"source_path": str(input_path), "ingest_run_id": make_id("ingest")},
            )
        )
    return chunks
