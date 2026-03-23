from __future__ import annotations

from pydantic import BaseModel, Field


class DatasetArtifact(BaseModel):
    path: str
    kind: str
    record_count: int | None = None
    notes: str | None = None


class DatasetManifest(BaseModel):
    manifest_id: str
    tournament: str | None = None
    role: str
    description: str
    artifacts: list[DatasetArtifact] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    recommended_uses: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
