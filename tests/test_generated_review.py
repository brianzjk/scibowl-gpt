import shutil
from pathlib import Path

from scibowl.generation_review.store import GeneratedQuestionReviewStore
from scibowl.schema.common import AnswerMode, Category, ModelInfo, QuestionType, Verdict
from scibowl.schema.dataset import GeneratedQuestionRunRecord
from scibowl.schema.generation import DraftQuestion, GeneratedDraft, QuestionSpec, RetrievalBundle
from scibowl.schema.review import HumanReview
from scibowl.schema.verification import ReviewMetadata, VerificationCheck, VerificationChecks, VerifierReport
from scibowl.utils.ids import make_id
from scibowl.utils.io import read_jsonl


def test_generated_question_review_store_round_trip() -> None:
    tmp_path = Path('tests_runtime') / make_id('generated_review')
    tmp_path.mkdir(parents=True, exist_ok=True)
    runs_path = tmp_path / 'runs.jsonl'
    reviews_path = tmp_path / 'reviews.jsonl'
    runs_path.write_text(_build_run_record('draft_1').model_dump_json() + '\n', encoding='utf-8')

    store = GeneratedQuestionReviewStore(runs_path, reviews_path, reviewer_id='brian')
    assert store.summary()['total'] == 1
    assert store.question_payload('draft_1')['metadata']['source_label'] == 'biology_test'

    updated = store.save_review(
        draft_id='draft_1',
        difficulty=5,
        quality=1.0,
        comment='Usable with edits.',
        verifier_scores={
            'format_compliance': 0.7,
            'factuality': 1.0,
            'topic_alignment': 0.9,
            'style_alignment': 0.8,
        },
        verifier_comment='This should not have been rejected.',
    )
    assert updated['review']['ratings']['difficulty'] == 5
    assert updated['review']['metadata']['human_verifier_scores']['style_alignment'] == 0.8

    saved_reviews = read_jsonl(reviews_path, HumanReview)
    assert len(saved_reviews) == 1
    assert saved_reviews[0].question_id == 'draft_1'
    assert saved_reviews[0].ratings.quality == 1.0

    cleared = store.save_review(
        draft_id='draft_1',
        difficulty=None,
        quality=None,
        comment=None,
        verifier_scores=None,
        verifier_comment=None,
    )
    assert cleared['review'] is None
    assert reviews_path.read_text(encoding='utf-8') == ''
    shutil.rmtree(tmp_path)


def _build_run_record(draft_id: str) -> GeneratedQuestionRunRecord:
    spec = QuestionSpec(
        spec_id='spec_1',
        category=Category.BIOLOGY,
        subcategory='cell biology',
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
        topic_focus=['mitochondria'],
    )
    draft = GeneratedDraft(
        draft_id=draft_id,
        spec_id=spec.spec_id,
        model_info=ModelInfo(provider='local', model_name='test-writer', prompt_version='writer_v1'),
        question=DraftQuestion(
            category=Category.BIOLOGY,
            subcategory='cell biology',
            question_type=QuestionType.TOSSUP,
            answer_mode=AnswerMode.SHORT_ANSWER,
            difficulty=4,
            question_text='Identify this organelle with its own DNA.',
            answer_text='ANSWER: mitochondrion',
        ),
    )
    report = VerifierReport(
        report_id='verify_1',
        draft_id=draft_id,
        spec_id=spec.spec_id,
        verdict=Verdict.PASS,
        summary='Looks acceptable.',
        checks=VerificationChecks(
            format_compliance=VerificationCheck(passed=True),
            factual_grounding=VerificationCheck(passed=True),
            answerability=VerificationCheck(passed=True),
            difficulty_alignment=VerificationCheck(passed=True, estimated_difficulty=4),
            topic_alignment=VerificationCheck(passed=True, style_score=0.6),
            style_alignment=VerificationCheck(passed=True, style_score=0.7),
            novelty=VerificationCheck(passed=True, similarity_score=0.2),
        ),
        review_metadata=ReviewMetadata(
            provider='local',
            model_name='test-verifier',
            prompt_version='verifier_v1',
        ),
    )
    return GeneratedQuestionRunRecord(
        run_id='generation_batch',
        job_id='biology_test',
        spec=spec,
        bundle=RetrievalBundle(spec_id=spec.spec_id),
        draft=draft,
        report=report,
    )
