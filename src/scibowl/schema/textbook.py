from __future__ import annotations

from pydantic import BaseModel, Field

from .common import SourceType


class TextbookChunk(BaseModel):
    chunk_id: str
    document_id: str
    source_type: SourceType = SourceType.TEXTBOOK
    title: str
    chapter: str | None = None
    section: str | None = None
    pages: list[int] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    text: str
    char_count: int
    token_count_est: int
    metadata: dict[str, object] = Field(default_factory=dict)
