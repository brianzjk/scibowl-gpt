from __future__ import annotations

from pydantic import BaseModel, Field


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
