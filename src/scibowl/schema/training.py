from __future__ import annotations

from pydantic import BaseModel, Field

from .common import AnswerMode, Category, QuestionType


class ChatMessage(BaseModel):
    role: str
    content: str


class SFTMetadata(BaseModel):
    source_question_id: str
    source_dataset: str
    source_tournament: str | None = None
    source_round_number: int | None = None
    category: Category
    subcategory: str
    question_type: QuestionType
    target_answer_mode: AnswerMode
    difficulty: int
    sample_weight: float = 1.0
    weight_reasons: list[str] = Field(default_factory=list)
    split: str
    repeat_count: int = 1
    has_specific_subcategory: bool = True
    dedupe_fingerprint: str | None = None


class SFTExample(BaseModel):
    example_id: str
    messages: list[ChatMessage]
    metadata: SFTMetadata
