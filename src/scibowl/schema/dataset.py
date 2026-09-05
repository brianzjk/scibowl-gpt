from __future__ import annotations

from pydantic import BaseModel, Field

from .generation import GeneratedDraft, QuestionSpec, RetrievalBundle
from .verification import VerifierReport


class GeneratedQuestionRunRecord(BaseModel):
    run_id: str
    job_id: str
    spec: QuestionSpec
    bundle: RetrievalBundle
    draft: GeneratedDraft
    report: VerifierReport
    metadata: dict[str, object] = Field(default_factory=dict)
