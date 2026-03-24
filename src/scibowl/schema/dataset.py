from __future__ import annotations

from pydantic import BaseModel, Field

from .generation import GeneratedDraft, QuestionSpec, RetrievalBundle
from .question import NormalizedQuestion
from .verification import VerifierReport


class TrainingExample(BaseModel):
    example_id: str
    task: str
    input: dict[str, object]
    output: dict[str, object]
    messages: list[dict[str, str]] = Field(default_factory=list)


class EvaluationRecord(BaseModel):
    run_id: str
    candidate_question_id: str
    spec_id: str
    scores: dict[str, float]
    human_review: dict[str, object] = Field(default_factory=dict)
    system_config: dict[str, object] = Field(default_factory=dict)


class BaselineRunRecord(BaseModel):
    run_id: str
    candidate_question_id: str
    spec: QuestionSpec
    draft: GeneratedDraft
    report: VerifierReport
    evaluation: EvaluationRecord
    reference_question: NormalizedQuestion
    reference_human_review: dict[str, object] = Field(default_factory=dict)


class GeneratedQuestionRunRecord(BaseModel):
    run_id: str
    job_id: str
    spec: QuestionSpec
    bundle: RetrievalBundle
    draft: GeneratedDraft
    report: VerifierReport
    evaluation: EvaluationRecord
    metadata: dict[str, object] = Field(default_factory=dict)
