from __future__ import annotations

from scibowl.generate.writer import WriterService
from scibowl.retrieval.search import retrieve_bundle
from scibowl.schema.generation import GeneratedDraft, QuestionSpec, RetrievalBundle
from scibowl.schema.question import NormalizedQuestion
from scibowl.schema.textbook import TextbookChunk
from scibowl.schema.verification import VerifierReport
from scibowl.verify.pipeline import VerifierService


class GenerationOrchestrator:
    def __init__(
        self,
        writer: WriterService | None = None,
        verifier: VerifierService | None = None,
    ) -> None:
        self.writer = writer or WriterService()
        self.verifier = verifier or VerifierService()

    def run(
        self,
        spec: QuestionSpec,
        textbook_chunks: list[TextbookChunk],
        style_questions: list[NormalizedQuestion],
        *,
        avoid_fact_chunk_ids: set[str] | None = None,
        avoid_style_question_ids: set[str] | None = None,
    ) -> tuple[RetrievalBundle, GeneratedDraft, VerifierReport]:
        bundle = retrieve_bundle(
            spec,
            textbook_chunks,
            style_questions,
            avoid_fact_chunk_ids=avoid_fact_chunk_ids,
            avoid_style_question_ids=avoid_style_question_ids,
        )
        draft = self.writer.write(spec, bundle)
        report = self.verifier.verify(spec, draft, bundle, style_questions)
        return bundle, draft, report
