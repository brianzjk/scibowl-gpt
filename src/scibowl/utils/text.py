from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def estimate_token_count(text: str) -> int:
    return max(1, len(text.split()))


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


def lexical_overlap_score(query_terms: set[str], text: str) -> float:
    text_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not query_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)
