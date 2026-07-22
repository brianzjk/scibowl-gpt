from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from scibowl.schema.textbook import TextbookChunk
from scibowl.utils.ids import make_id, slugify
from scibowl.utils.text import estimate_token_count, normalize_whitespace, repair_mojibake


DEFAULT_CORPUS_VERSION = 'textbook_v2'


@dataclass(frozen=True)
class _PageWordChunk:
    text: str
    pages: list[int]
    chapter: str | None
    section: str | None


def _read_textbook_pages(
    input_path: Path,
    *,
    start_page: int | None,
    end_page: int | None,
) -> tuple[list[str], int, int]:
    if input_path.suffix.lower() == '.pdf':
        reader = PdfReader(str(input_path))
        page_count = len(reader.pages)
        first_page, last_page = _validated_page_bounds(
            page_count,
            start_page=start_page,
            end_page=end_page,
        )
        pages = [
            reader.pages[index].extract_text() or ''
            for index in range(first_page - 1, last_page)
        ]
        return pages, first_page, page_count

    first_page, last_page = _validated_page_bounds(
        1,
        start_page=start_page,
        end_page=end_page,
    )
    return [input_path.read_text(encoding='utf-8')], first_page, last_page


def ingest_textbook_text(
    input_path: Path,
    document_id: str | None = None,
    title: str | None = None,
    topics: list[str] | None = None,
    *,
    start_page: int | None = None,
    end_page: int | None = None,
    max_words: int = 180,
    overlap_words: int = 30,
    corpus_version: str = DEFAULT_CORPUS_VERSION,
    expected_sha256: str | None = None,
    expected_page_count: int | None = None,
    ingest_run_id: str | None = None,
) -> list[TextbookChunk]:
    source_sha256 = _file_sha256(input_path)
    if expected_sha256 is not None and source_sha256.casefold() != expected_sha256.casefold():
        raise ValueError(
            f'SHA-256 mismatch for {input_path}: expected {expected_sha256}, found {source_sha256}'
        )
    pages, extracted_first_page, page_count = _read_textbook_pages(
        input_path,
        start_page=start_page,
        end_page=end_page,
    )
    if expected_page_count is not None and page_count != expected_page_count:
        raise ValueError(
            f'page count mismatch for {input_path}: expected {expected_page_count}, found {page_count}'
        )
    if start_page is not None or end_page is not None:
        selected_pages = pages
        first_page = extracted_first_page
        last_page = first_page + len(selected_pages) - 1
        trim_metadata = {
            'manual_page_bounds': True,
            'content_start_page': first_page,
            'content_end_page': last_page,
            'front_matter_trimmed': first_page > 1,
            'back_matter_trimmed': last_page < page_count,
        }
    else:
        selected_pages, first_page, trim_metadata = _select_content_pages(
            pages,
            start_page=None,
            end_page=None,
        )
    doc_id = document_id or slugify(input_path.stem)
    doc_title = title or input_path.stem
    topic_list = topics or []
    run_id = ingest_run_id or make_id('ingest')

    chunks: list[TextbookChunk] = []
    page_chunks = _chunk_selected_pages(
        selected_pages,
        first_page=first_page,
        max_words=max_words,
        overlap_words=overlap_words,
    )
    for chunk_index, page_chunk in enumerate(page_chunks, start=1):
        start_page = page_chunk.pages[0]
        end_page = page_chunk.pages[-1]
        content_hash = hashlib.sha1(
            f'{doc_id}|{page_chunk.pages}|{page_chunk.text}'.encode('utf-8')
        ).hexdigest()[:12]
        page_span = f'p{start_page:04d}' if start_page == end_page else f'p{start_page:04d}-{end_page:04d}'
        chunk_id = f'{doc_id}_{page_span}_{chunk_index:05d}_{content_hash}'
        chunks.append(
            TextbookChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                title=doc_title,
                chapter=page_chunk.chapter,
                section=page_chunk.section,
                pages=page_chunk.pages,
                topics=topic_list,
                text=page_chunk.text,
                char_count=len(page_chunk.text),
                token_count_est=estimate_token_count(page_chunk.text),
                metadata={
                    'source_path': str(input_path),
                    'source_sha256': source_sha256,
                    'ingest_run_id': run_id,
                    'corpus_version': corpus_version,
                    'chunk_index': chunk_index,
                    **trim_metadata,
                },
            )
        )
    return chunks


def _validated_page_bounds(
    page_count: int,
    *,
    start_page: int | None,
    end_page: int | None,
) -> tuple[int, int]:
    if page_count <= 0:
        if start_page is not None or end_page is not None:
            raise ValueError('page bounds cannot be used with an empty source')
        return 1, 0
    if start_page is not None and not 1 <= start_page <= page_count:
        raise ValueError('start_page must be within the source')
    if end_page is not None and not 1 <= end_page <= page_count:
        raise ValueError('end_page must be within the source')
    first_page = start_page or 1
    last_page = end_page or page_count
    if first_page > last_page:
        raise ValueError('start_page must not be after end_page')
    return first_page, last_page


def _chunk_selected_pages(
    pages: list[str],
    *,
    first_page: int,
    max_words: int,
    overlap_words: int,
) -> list[_PageWordChunk]:
    if max_words <= 0:
        raise ValueError('max_words must be positive')
    if overlap_words < 0:
        raise ValueError('overlap_words must not be negative')
    if overlap_words >= max_words:
        raise ValueError('overlap_words must be less than max_words')

    word_records: list[tuple[str, int, str | None, str | None]] = []
    current_chapter: str | None = None
    current_section: str | None = None
    for page_offset, raw_page in enumerate(pages):
        page_number = first_page + page_offset
        current_chapter, current_section = _extract_page_headings(
            raw_page,
            current_chapter=current_chapter,
            current_section=current_section,
        )
        cleaned_page = _clean_textbook_page(raw_page)
        for word in cleaned_page.split():
            word_records.append((word, page_number, current_chapter, current_section))

    if not word_records:
        return []

    chunks: list[_PageWordChunk] = []
    step = max_words - overlap_words
    for start in range(0, len(word_records), step):
        records = word_records[start:start + max_words]
        if not records:
            break
        chunk_pages = sorted({record[1] for record in records})
        chapter = next((record[2] for record in records if record[2] is not None), None)
        section = next((record[3] for record in records if record[3] is not None), None)
        chunks.append(
            _PageWordChunk(
                text=' '.join(record[0] for record in records),
                pages=chunk_pages,
                chapter=chapter,
                section=section,
            )
        )
        if start + max_words >= len(word_records):
            break
    return chunks


def _select_content_pages(
    pages: list[str],
    *,
    start_page: int | None,
    end_page: int | None,
) -> tuple[list[str], int, dict[str, object]]:
    if not pages and (start_page is not None or end_page is not None):
        raise ValueError('page bounds cannot be used with an empty source')
    if not pages:
        return [], 1, {'front_matter_trimmed': False, 'back_matter_trimmed': False}

    if start_page is not None or end_page is not None:
        if start_page is not None and not 1 <= start_page <= len(pages):
            raise ValueError('start_page must be within the source')
        if end_page is not None and not 1 <= end_page <= len(pages):
            raise ValueError('end_page must be within the source')
        first_page = start_page or 1
        last_page = end_page or len(pages)
        if first_page > last_page:
            raise ValueError('start_page must not be after end_page')
        return pages[first_page - 1:last_page], first_page, {
            'manual_page_bounds': True,
            'content_start_page': first_page,
            'content_end_page': last_page,
            'front_matter_trimmed': first_page > 1,
            'back_matter_trimmed': last_page < len(pages),
        }

    front_trimmed, front_metadata = _trim_pages_after_contents(pages)
    first_page = int(front_metadata.get('content_start_page') or 1)
    back_trimmed, back_metadata = _trim_pages_before_back_matter(front_trimmed)
    if back_metadata.get('back_matter_trimmed'):
        relative_start = int(back_metadata['back_matter_start_page'])
        back_metadata['back_matter_start_page'] = first_page + relative_start - 1
    return back_trimmed, first_page, {
        'manual_page_bounds': False,
        'content_end_page': first_page + len(back_trimmed) - 1,
        **front_metadata,
        **back_metadata,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _extract_page_headings(
    text: str,
    *,
    current_chapter: str | None,
    current_section: str | None,
) -> tuple[str | None, str | None]:
    text = repair_mojibake(text)
    chapter = current_chapter
    section = current_section
    lines = [normalize_whitespace(line) for line in text.splitlines() if normalize_whitespace(line)]
    for line in lines[:8]:
        chapter_match = re.match(
            r'^(?:\d+\s*)?CHAPTER\s+([0-9]+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\b',
            line,
        )
        if chapter_match and len(line.split()) <= 12 and '.' not in line:
            next_chapter = f'Chapter {chapter_match.group(1).title()}'
            if next_chapter != chapter:
                chapter = next_chapter
                section = None
            continue
        section_match = re.match(
            r'^(?:CONCEPT\s+)?([1-9][0-9]?(?:[.-][0-9]{1,2}){1,2})\s+(.+)$',
            line,
        )
        if section_match and _looks_like_heading_title(section_match.group(2)):
            section = f'{section_match.group(1)} {section_match.group(2)}'
    return chapter, section


def _looks_like_heading_title(value: str) -> bool:
    if len(value.split()) > 12 or value.endswith(('.', '?', '!')):
        return False
    first_letter = next((character for character in value if character.isalpha()), '')
    return bool(first_letter and first_letter.isupper())


def _clean_textbook_page(text: str) -> str:
    text = repair_mojibake(text)
    text = _strip_inline_chapter_contents(text)
    compact = normalize_whitespace(text).casefold()
    if _is_review_page(compact) or _is_review_continuation_page(text):
        return ''

    cleaned_lines: list[str] = []
    prompt_lines_to_skip = 0

    for raw_line in text.splitlines():
        line = normalize_whitespace(raw_line)
        if not line:
            if cleaned_lines and cleaned_lines[-1] != '':
                cleaned_lines.append('')
            continue

        lowered = line.casefold()
        if _starts_embedded_prompt(lowered):
            prompt_lines_to_skip = 2
            continue
        if prompt_lines_to_skip:
            prompt_lines_to_skip -= 1
            continue
        if _is_page_furniture(line, lowered):
            continue
        if _starts_review_section(lowered):
            break
        if _starts_exercise_block(line, lowered):
            continue
        if _is_exercise_or_review_line(line, lowered):
            continue

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()


def _strip_inline_chapter_contents(text: str) -> str:
    compact = normalize_whitespace(text)
    match = re.search(
        r'\bsummary\s+questions for review\s+questions for thought and exploration\s+contents\b',
        compact,
        flags=re.IGNORECASE,
    )
    if match is None:
        return text
    tail = compact[match.end():].strip()
    return tail if len(tail.split()) >= 30 else text


def _is_review_page(compact: str) -> bool:
    return any(
        marker in compact
        for marker in (
            'chapter review',
            'for selected answers',
            'questions for review',
            'end-of-chapter questions',
        )
    )


def _starts_embedded_prompt(lowered: str) -> bool:
    return bool(
        re.match(
            r'^(visual skills|interpret the data|make connections|draw it|what if\?|evolution connection)\b',
            lowered,
        )
    )


def _is_review_continuation_page(text: str) -> bool:
    numbered_questions = sum(
        bool(re.match(r'^\s*(?:\d{1,2}|[IVX]{1,3})[.)]\s+', line, flags=re.IGNORECASE))
        for line in text.splitlines()
    )
    return numbered_questions >= 4 and text.count('?') >= 3


def _starts_review_section(lowered: str) -> bool:
    if 'test your understanding' in lowered:
        return True
    return bool(
        re.match(
            r'^(questions|chapter review|review questions|end-of-chapter questions|problems|exercises)\b',
            lowered,
        )
    )


def _trim_pages_after_contents(pages: list[str]) -> tuple[list[str], dict[str, object]]:
    if not pages:
        return [], {'front_matter_trimmed': False}

    search_limit = min(len(pages), 50)
    contents_index = next(
        (index for index in range(search_limit) if _has_contents_heading(pages[index])),
        None,
    )
    if contents_index is None:
        return pages, {'front_matter_trimmed': False, 'content_start_page': 1}

    chapter_start = next(
        (
            index
            for index in range(contents_index + 1, min(len(pages), contents_index + 40))
            if _looks_like_first_chapter_page(pages[index])
        ),
        None,
    )
    if chapter_start is None:
        end_index = contents_index
        for index in range(contents_index, min(len(pages), contents_index + 20)):
            if _looks_like_contents_page(pages[index]):
                end_index = index
            elif index > contents_index:
                break
        chapter_start = min(end_index + 1, len(pages) - 1)

    return pages[chapter_start:], {
        'front_matter_trimmed': True,
        'contents_start_page': contents_index + 1,
        'content_start_page': chapter_start + 1,
    }


def _trim_pages_before_back_matter(pages: list[str]) -> tuple[list[str], dict[str, object]]:
    if not pages:
        return [], {'back_matter_trimmed': False}

    search_start = max(0, len(pages) - 100)
    for index in range(search_start, len(pages)):
        if _has_back_matter_heading(pages[index]):
            return pages[:index], {
                'back_matter_trimmed': True,
                'back_matter_start_page': index + 1,
            }
    return pages, {'back_matter_trimmed': False}


def _has_contents_heading(text: str) -> bool:
    for raw_line in text.splitlines()[:20]:
        line = normalize_whitespace(raw_line).casefold().strip(':')
        if line in {'contents', 'table of contents'}:
            return True
    return False


def _looks_like_first_chapter_page(text: str) -> bool:
    if _looks_like_contents_page(text):
        return False
    lines = [normalize_whitespace(line) for line in text.splitlines() if normalize_whitespace(line)]
    has_chapter_one = any(re.match(r'^chapter\s+1\b', line, flags=re.IGNORECASE) for line in lines[:20])
    return has_chapter_one and len(normalize_whitespace(text).split()) >= 80


def _looks_like_contents_page(text: str) -> bool:
    if _has_contents_heading(text):
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    dotted_lines = sum('...' in line for line in lines)
    indexed_lines = sum(
        1
        for line in lines
        if re.search(r'(chapter|appendix|\d+(\.\d+)*)', line.lower()) and re.search(r'\d+\s*$', line)
    )
    return dotted_lines >= 2 or indexed_lines >= 5


def _has_back_matter_heading(text: str) -> bool:
    headings = {
        'glossary',
        'index',
        'selected answers',
        'answer key',
        'bibliography',
        'photo credits',
    }
    lines = [normalize_whitespace(line).casefold().strip(':') for line in text.splitlines() if normalize_whitespace(line)]
    return any(line in headings for line in lines[:20])


def _starts_exercise_block(line: str, lowered: str) -> bool:
    patterns = (
        r'^(problem|problems)\s+\d',
        r'^(concept checks?|review questions?|discussion questions?|key terms?)\b',
        r'^(exercises?|chapter review)\b',
        r'^q\s',
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def _is_exercise_or_review_line(line: str, lowered: str) -> bool:
    patterns = (
        r'^\(?[a-z]\)\s',
        r'^(hints?|answer|answers)\b',
    )
    if any(re.search(pattern, lowered) for pattern in patterns):
        return True
    return 'key terms' in lowered or 'concept checks' in lowered


def _is_page_furniture(line: str, lowered: str) -> bool:
    if re.fullmatch(r'\d+', line):
        return True
    if 'indd' in lowered:
        return True
    if re.search(r'\b(am|pm)\b', lowered) and re.search(r'\d+/\d+/\d+', lowered):
        return True
    if '|' in line and ('chapter' in lowered or 'part' in lowered):
        return True
    return False
