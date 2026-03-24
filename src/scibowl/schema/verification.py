from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from .common import Verdict


class VerificationIssue(BaseModel):
    code: str
    message: str
    span: str | None = None
    citations_checked: list[str] = Field(default_factory=list)


class VerificationCheck(BaseModel):
    passed: bool
    issues: list[VerificationIssue] = Field(default_factory=list)
    estimated_difficulty: int | None = None
    style_score: float | None = None
    similarity_score: float | None = None
    bait_score: float | None = None
    sampled_prefixes: list[dict[str, object]] = Field(default_factory=list)


class VerificationChecks(BaseModel):
    format_compliance: VerificationCheck
    factual_grounding: VerificationCheck
    answerability: VerificationCheck
    difficulty_alignment: VerificationCheck
    style_alignment: VerificationCheck
    novelty: VerificationCheck
    bait_detection: VerificationCheck


class ReviewMetadata(BaseModel):
    provider: str
    model_name: str
    prompt_version: str
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VerifierReport(BaseModel):
    report_id: str
    draft_id: str
    spec_id: str
    verdict: Verdict
    summary: str
    checks: VerificationChecks
    required_revisions: list[str] = Field(default_factory=list)
    review_metadata: ReviewMetadata
