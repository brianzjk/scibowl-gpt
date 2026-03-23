from __future__ import annotations

from scibowl.generate.local_model import HeuristicWriterModel, PromptWriterModel, build_writer_model
from scibowl.schema.generation import GeneratedDraft, QuestionSpec, RetrievalBundle


class WriterService:
    def __init__(self, model: PromptWriterModel | HeuristicWriterModel | None = None) -> None:
        self.model = model or build_writer_model()

    def write(self, spec: QuestionSpec, bundle: RetrievalBundle) -> GeneratedDraft:
        return self.model.generate(spec, bundle)
