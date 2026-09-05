from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator

from .common import AnswerMode, Category, Citation, ModelInfo, QuestionType
from .question import Choice


class QuestionConstraints(BaseModel):
    max_length_words: int = 90
    require_single_unambiguous_answer: bool = True
    disallow_multi_step_calculation: bool = False
    allow_negative_questions: bool = False


class QuestionSpec(BaseModel):
    spec_id: str
    category: Category
    subcategory: str
    question_type: QuestionType
    answer_mode: AnswerMode | None = None
    difficulty: int
    topic_focus: list[str] = Field(default_factory=list)
    must_use_sources: list[str] = Field(default_factory=list)
    forbidden_topics: list[str] = Field(default_factory=list)
    constraints: QuestionConstraints = Field(default_factory=QuestionConstraints)

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, value: int) -> int:
        if not 1 <= value <= 7:
            raise ValueError("difficulty must be between 1 and 7")
        return value


class RetrievedFactChunk(BaseModel):
    chunk_id: str
    document_id: str
    locator: str | None = None
    text: str


class RetrievedStyleExample(BaseModel):
    question_id: str
    question_text: str
    answer_text: str
    difficulty: int
    question_type: QuestionType
    answer_mode: AnswerMode


class RetrievalBundle(BaseModel):
    spec_id: str
    fact_chunks: list[RetrievedFactChunk] = Field(default_factory=list)
    style_examples: list[RetrievedStyleExample] = Field(default_factory=list)


class DraftQuestion(BaseModel):
    category: Category
    subcategory: str
    question_type: QuestionType
    answer_mode: AnswerMode
    difficulty: int
    question_text: str
    answer_text: str
    choices: list[Choice] = Field(default_factory=list)


class GenerationMetadata(BaseModel):
    temperature: float = 0.0
    seed: int = 0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GeneratedDraft(BaseModel):
    draft_id: str
    spec_id: str
    model_info: ModelInfo
    question: DraftQuestion
    citations: list[Citation] = Field(default_factory=list)
    generation_metadata: GenerationMetadata = Field(default_factory=GenerationMetadata)
