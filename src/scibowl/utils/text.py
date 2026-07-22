from __future__ import annotations

import re


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
    if not paragraphs:
        return []

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
            if overlap + previous_words > overlap_words and next_start < end_index:
                break
            overlap += previous_words
            next_start -= 1
            if overlap >= overlap_words:
                break
        if next_start == start_index:
            next_start = min(start_index + 1, len(paragraphs))
        start_index = next_start

    return chunks


def lexical_overlap_score(query_terms: set[str], text: str) -> float:
    text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not query_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)
