from __future__ import annotations

import re
import unicodedata


_MOJIBAKE_SEQUENCE = re.compile(r'(?:Ã.|Â.|â..|ï..)')


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8-as-Windows-1252 PDF extraction artifacts."""

    def decode_match(match: re.Match[str]) -> str:
        value = match.group(0)
        try:
            return value.encode('cp1252').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return value

    repaired = _MOJIBAKE_SEQUENCE.sub(decode_match, text)
    return unicodedata.normalize('NFKC', repaired)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def estimate_token_count(text: str) -> int:
    return max(1, len(text.split()))


def normalize_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n+", text):
        normalized = normalize_whitespace(paragraph)
        if normalized:
            paragraphs.append(normalized)
    return paragraphs


def chunk_text(text: str, max_words: int = 160, overlap_words: int = 30) -> list[str]:
    _validate_chunk_sizes(max_words, overlap_words)
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, max_words - overlap_words)
    for start in range(0, len(words), step):
        end = start + max_words
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks


def chunk_paragraphs(paragraphs: list[str], max_words: int = 160, overlap_words: int = 30) -> list[str]:
    _validate_chunk_sizes(max_words, overlap_words)
    if not paragraphs:
        return []

    bounded_paragraphs: list[str] = []
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        if len(paragraph.split()) <= max_words:
            bounded_paragraphs.append(paragraph)
        else:
            bounded_paragraphs.extend(
                chunk_text(paragraph, max_words=max_words, overlap_words=overlap_words)
            )

    paragraphs = bounded_paragraphs

    chunks: list[str] = []
    start_index = 0
    while start_index < len(paragraphs):
        current_words = 0
        end_index = start_index
        selected: list[str] = []
        while end_index < len(paragraphs):
            paragraph = paragraphs[end_index]
            paragraph_words = len(paragraph.split())
            if selected and current_words + paragraph_words > max_words:
                break
            selected.append(paragraph)
            current_words += paragraph_words
            end_index += 1
            if current_words >= max_words:
                break

        if not selected:
            selected = [paragraphs[start_index]]
            end_index = start_index + 1

        chunks.append(" ".join(selected).strip())
        if end_index >= len(paragraphs):
            break

        overlap = 0
        next_start = end_index
        while next_start > start_index:
            previous_words = len(paragraphs[next_start - 1].split())
            if overlap + previous_words > overlap_words:
                break
            overlap += previous_words
            next_start -= 1
            if overlap >= overlap_words:
                break
        start_index = end_index if next_start == start_index else next_start

    return chunks


def _validate_chunk_sizes(max_words: int, overlap_words: int) -> None:
    if max_words <= 0:
        raise ValueError('max_words must be positive')
    if overlap_words < 0:
        raise ValueError('overlap_words must not be negative')
    if overlap_words >= max_words:
        raise ValueError('overlap_words must be less than max_words')


def lexical_overlap_score(query_terms: set[str], text: str) -> float:
    text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not query_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)
