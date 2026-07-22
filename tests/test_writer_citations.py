from scibowl.generate.local_model import PromptWriterModel, build_generated_draft
from scibowl.schema.common import AnswerMode, Category, ModelInfo, QuestionType
from scibowl.schema.generation import QuestionSpec, RetrievalBundle, RetrievedFactChunk


def _spec() -> QuestionSpec:
    return QuestionSpec(
        spec_id='citation_spec',
        category=Category.BIOLOGY,
        subcategory='Cell Biology',
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
        topic_focus=['mitochondria'],
    )


def _bundle() -> RetrievalBundle:
    return RetrievalBundle(
        spec_id='citation_spec',
        fact_chunks=[
            RetrievedFactChunk(
                chunk_id='used',
                document_id='campbell',
                locator='Campbell; p. 12',
                text='Mitochondria make ATP through oxidative phosphorylation.',
            ),
            RetrievedFactChunk(
                chunk_id='unused',
                document_id='campbell',
                locator='Campbell; p. 20',
                text='Chloroplasts carry out photosynthesis.',
            ),
        ],
    )


def test_build_generated_draft_cites_only_reported_chunks() -> None:
    draft = build_generated_draft(
        spec=_spec(),
        bundle=_bundle(),
        model_info=ModelInfo(provider='test', model_name='test', prompt_version='test'),
        question_text='Which organelle makes most cellular ATP?',
        answer_text='ANSWER: mitochondrion',
        choices=[],
        citation_chunk_ids=['used', 'missing'],
    )

    assert [citation.chunk_id for citation in draft.citations] == ['used']


def test_prompt_writer_reads_citation_ids_from_model_output() -> None:
    class Client:
        def complete_json(self, **kwargs):
            return {
                'question_text': 'Which organelle makes most cellular ATP?',
                'answer_text': 'ANSWER: mitochondrion',
                'choices': [],
                'citation_chunk_ids': ['used'],
            }

    draft = PromptWriterModel(client=Client(), model_name='test').generate(_spec(), _bundle())

    assert [citation.chunk_id for citation in draft.citations] == ['used']
