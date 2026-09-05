from __future__ import annotations

from pydantic import BaseModel, Field

from .common import AnswerMode, Category, QuestionType


class ChatMessage(BaseModel):
    role: str
    content: str


class CleanSFTMetadata(BaseModel):
    source_question_id: str
    duplicate_source_question_ids: list[str] = Field(default_factory=list)
    source_dataset: str
    source_family: str
    source_year: int | None = None
    category: Category
    subcategory: str | None = None
    subcategory_source: str | None = None
    question_type: QuestionType
    target_answer_mode: AnswerMode
    difficulty: int | None = None
    difficulty_mean: float | None = None
    difficulty_source: str | None = None
    quality_mean: float | None = None
    quality_source: str | None = None
    profile: str
    split: str
    duplicate_group: str
    exact_fingerprint: str
    near_fingerprint: str
    sample_weight: float = 1.0


class CleanSFTExample(BaseModel):
    example_id: str
    messages: list[ChatMessage]
    metadata: CleanSFTMetadata
