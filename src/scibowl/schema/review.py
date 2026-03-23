from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


class ReviewRatings(BaseModel):
    quality: float | None = None
    difficulty: float | None = None


class HumanReview(BaseModel):
    review_id: str
    question_id: str
    reviewer_id: str
    tournament: str | None = None
    review_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ratings: ReviewRatings = Field(default_factory=ReviewRatings)
    comment: str | None = None
    status: str | None = None
    source_row: int | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("reviewer_id")
    @classmethod
    def validate_reviewer_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("reviewer_id must not be empty")
        return value
