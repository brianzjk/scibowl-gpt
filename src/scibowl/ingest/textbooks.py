from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from scibowl.schema.textbook import TextbookChunk
from scibowl.utils.ids import make_id, slugify
from scibowl.utils.text import chunk_paragraphs, estimate_token_count, normalize_paragraphs, normalize_whitespace


def _read_textbook_pages(input_path: Path) -> list[str]:
    if input_path.suffix.lower() == ".pdf":
        reader = PdfReader(str(input_path))
        return [page.extract_text() or "" for page in reader.pages]
    return [input_path.read_text(encoding="utf-8")]


def ingest_textbook_text(
    input_path: Path,
    document_id: str | None = None,
    title: str | None = None,
    topics: list[str] | None = None,
) -> list[TextbookChunk]:
    pages = _read_textbook_pages(input_path)
    trimmed_pages, trim_metadata = _trim_pages_after_contents(pages)
    trimmed_pages, back_matter_metadata = _trim_pages_before_back_matter(trimmed_pages)
    cleaned_pages = [_clean_textbook_page(page) for page in trimmed_pages]
    raw_text = "\n\n".join(page for page in cleaned_pages if page.strip())
    paragraphs = normalize_paragraphs(raw_text)
    doc_id = document_id or slugify(input_path.stem)
    doc_title = title or input_path.stem
    topic_list = topics or []

    chunks: list[TextbookChunk] = []
    for index, chunk in enumerate(chunk_paragraphs(paragraphs), start=1):
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
                metadata={
                    "source_path": str(input_path),
                    "ingest_run_id": make_id("ingest"),
                    **trim_metadata,
                    **back_matter_metadata,
                },
            )
        )
    return chunks


def _clean_textbook_page(text: str) -> str:
    cleaned_lines: list[str] = []
    skip_until_blank = False

    for raw_line in text.splitlines():
        line = normalize_whitespace(raw_line)
        if not line:
            skip_until_blank = False
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        lowered = line.casefold()
        if skip_until_blank:
            continue
        if _is_page_furniture(line, lowered):
            continue
        if _starts_exercise_block(line, lowered):
            skip_until_blank = True
            continue
        if _is_exercise_or_review_line(line, lowered):
            continue

        cleaned_lines.append(line)

    return "\n".join(line for line in cleaned_lines if line or cleaned_lines)


def _trim_pages_after_contents(pages: list[str]) -> tuple[list[str], dict[str, object]]:
    if not pages:
        return [], {"front_matter_trimmed": False}

    search_limit = min(len(pages), 40)
    start_idx: int | None = None
    for index in range(search_limit):
        if _has_contents_heading(pages[index]):
            start_idx = index
            break

    if start_idx is None:
        return pages, {"front_matter_trimmed": False}

    end_idx = start_idx
    for index in range(start_idx, min(len(pages), start_idx + 10)):
        if _looks_like_contents_page(pages[index]):
            end_idx = index
            continue
        if index > start_idx:
            break

    content_start = min(end_idx + 1, len(pages) - 1)
    return pages[content_start:], {
        "front_matter_trimmed": True,
        "contents_start_page": start_idx + 1,
        "content_start_page": content_start + 1,
    }


def _trim_pages_before_back_matter(pages: list[str]) -> tuple[list[str], dict[str, object]]:
    if not pages:
        return [], {"back_matter_trimmed": False}

    search_start = max(0, len(pages) - 80)
    for index in range(search_start, len(pages)):
        if _has_back_matter_heading(pages[index]):
            return pages[:index], {
                "back_matter_trimmed": True,
                "back_matter_start_page": index + 1,
            }
    return pages, {"back_matter_trimmed": False}


def _has_contents_heading(text: str) -> bool:
    lowered = text.lower()
    return "table of contents" in lowered or bool(re.search(r"\bcontents\b", lowered))


def _looks_like_contents_page(text: str) -> bool:
    lowered = text.lower()
    if _has_contents_heading(text):
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    dotted_lines = sum("..." in line for line in lines)
    indexed_lines = sum(
        1
        for line in lines
        if re.search(r"(chapter|appendix|\d+(\.\d+)*)", line.lower()) and re.search(r"\d+\s*$", line)
    )
    return dotted_lines >= 2 or indexed_lines >= 3


def _has_back_matter_heading(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "glossary",
        "index",
        "selected answers",
        "answers to",
        "answer key",
        "references",
        "bibliography",
        "photo credits",
        "credits",
        "appendix",
        "appendices",
    )
    return any(re.search(rf"\b{re.escape(marker)}\b", lowered) for marker in markers)


def _starts_exercise_block(line: str, lowered: str) -> bool:
    patterns = (
        r"^(problem|problems)\s+\d",
        r"^(concept checks?|review questions?|discussion questions?|key terms?)\b",
        r"^(exercises?|chapter review|chapter summary|summary)\b",
        r"^q\s",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def _is_exercise_or_review_line(line: str, lowered: str) -> bool:
    patterns = (
        r"^\d+\.\s",
        r"^\(?[a-z]\)\s",
        r"^(hints?|answer|answers)\b",
    )
    if any(re.search(pattern, lowered) for pattern in patterns):
        return True
    if "key terms" in lowered or "concept checks" in lowered:
        return True
    return False


def _is_page_furniture(line: str, lowered: str) -> bool:
    if re.fullmatch(r"\d+", line):
        return True
    if "indd" in lowered:
        return True
    if re.search(r"\b(am|pm)\b", lowered) and re.search(r"\d+/\d+/\d+", lowered):
        return True
    if re.search(r"^chapter\s+\d+", lowered):
        return True
    if re.search(r"^part\s+\d+", lowered):
        return True
    if re.search(r"^\d+\s+chapter\b", lowered):
        return True
    if "|" in line and ("chapter" in lowered or "part" in lowered):
        return True
    if line.isupper() and len(line.split()) <= 8:
        return True
    return False
