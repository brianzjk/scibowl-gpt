import json
import shutil
from pathlib import Path

from scibowl.eval.baseline import run_baseline_eval
from scibowl.generation_review.store import GeneratedQuestionReviewStore
from scibowl.schema.common import AnswerMode, Category, ModelInfo, QuestionType, SourceType, Verdict
from scibowl.schema.dataset import BaselineRunRecord, EvaluationRecord, GeneratedQuestionRunRecord
from scibowl.schema.generation import DraftQuestion, GeneratedDraft, QuestionSpec, RetrievalBundle
from scibowl.schema.question import NormalizedQuestion, SourceMetadata
from scibowl.schema.review import HumanReview
from scibowl.schema.verification import ReviewMetadata, VerificationCheck, VerificationChecks, VerifierReport
from scibowl.utils.ids import make_id
from scibowl.utils.io import read_jsonl


def test_generated_question_review_store_round_trip() -> None:
    tmp_path = Path("tests_runtime") / make_id("generated_review")
    tmp_path.mkdir(parents=True, exist_ok=True)
    runs_path = tmp_path / "runs.jsonl"
    reviews_path = tmp_path / "reviews.jsonl"

    runs_path.write_text(_build_run_record("draft_1").model_dump_json() + "\n", encoding="utf-8")

    store = GeneratedQuestionReviewStore(runs_path, reviews_path, reviewer_id="brian")
    assert store.summary()["total"] == 1
    assert store.summary()["reviewed"] == 0
    payload = store.question_payload("draft_1")
    assert payload["metadata"]["source_label"] == "MIT 2025"

    updated = store.save_review(
        draft_id="draft_1",
        difficulty=5,
        quality=1.0,
        comment="Usable with edits.",
        verifier_scores={"format_compliance": 0.7, "factuality": 1.0, "topic_alignment": 0.9, "style_alignment": 0.8},
        verifier_comment="This should not have been rejected.",
    )
    assert updated["review"]["ratings"]["difficulty"] == 5
    assert updated["review"]["metadata"]["human_verifier_scores"]["format_compliance"] == 0.7
    assert updated["review"]["metadata"]["human_verifier_scores"]["style_alignment"] == 0.8
    assert updated["review"]["metadata"]["human_verifier_scores"]["factuality"] == 1.0
    assert updated["review"]["metadata"]["human_verifier_scores"]["topic_alignment"] == 0.9
    assert updated["display_scores"]["format_compliance"] == 1.0
    assert updated["display_scores"]["factuality"] == 1.0
    assert updated["display_scores"]["topic_alignment"] == 0.6
    assert updated["display_scores"]["style_alignment"] == 0.7

    saved_reviews = read_jsonl(reviews_path, HumanReview)
    assert len(saved_reviews) == 1
    assert saved_reviews[0].question_id == "draft_1"
    assert saved_reviews[0].ratings.quality == 1.0
    assert saved_reviews[0].metadata["human_verifier_scores"]["format_compliance"] == 0.7
    assert saved_reviews[0].metadata["human_verifier_scores"]["topic_alignment"] == 0.9
    assert saved_reviews[0].metadata["human_verifier_scores"]["style_alignment"] == 0.8

    cleared = store.save_review(
        draft_id="draft_1",
        difficulty=None,
        quality=None,
        comment=None,
        verifier_scores=None,
        verifier_comment=None,
    )
    assert cleared["review"] is None
    assert reviews_path.read_text(encoding="utf-8") == ""

    shutil.rmtree(tmp_path)


def test_run_baseline_eval_writes_detailed_runs(monkeypatch) -> None:
    tmp_path = Path("tests_runtime") / make_id("baseline_runs")
    tmp_path.mkdir(parents=True, exist_ok=True)
    questions_path = tmp_path / "questions.jsonl"
    reviews_path = tmp_path / "reviews.jsonl"
    split_path = tmp_path / "split.json"
    output_dir = tmp_path / "output"

    question = {
        "question_id": "q1",
        "source_type": "dataset",
        "source_id": "mit_2025",
        "category": "biology",
        "subcategory": "cell biology",
        "question_type": "tossup",
        "answer_mode": "short_answer",
        "difficulty": 4,
        "question_text": "What organelle contains mitochondrial DNA?",
        "answer_text": "ANSWER: mitochondrion",
        "source_metadata": {"round": 2, "tournament": "MIT 2025"},
    }
    questions_path.write_text(json.dumps(question) + "\n", encoding="utf-8")
    reviews_path.write_text(
        json.dumps(
            {
                "review_id": "r1",
                "question_id": "q1",
                "reviewer_id": "agg",
                "ratings": {"quality": 1.0, "difficulty": 4},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    split_path.write_text(
        json.dumps(
            {
                "question_set_path": str(questions_path),
                "reviews_path": str(reviews_path),
                "train_question_ids": [],
                "val_question_ids": [],
                "test_question_ids": ["q1"],
            }
        ),
        encoding="utf-8",
    )

    class FakeOrchestrator:
        def run(self, spec, textbook_chunks, style_questions, **kwargs):
            draft = _build_run_record("draft_fake", candidate_question_id="q1").draft
            report = _build_run_record("draft_fake", candidate_question_id="q1").report
            return RetrievalBundle(spec_id=spec.spec_id), draft, report

    monkeypatch.setattr("scibowl.eval.baseline.GenerationOrchestrator", FakeOrchestrator)

    records = run_baseline_eval(
        split_path=split_path,
        split_name="test",
        output_dir=output_dir,
        questions_path=questions_path,
        reviews_path=reviews_path,
    )
    assert len(records) == 1

    eval_records = read_jsonl(output_dir / "test_baseline_eval.jsonl", EvaluationRecord)
    run_records = read_jsonl(output_dir / "test_baseline_runs.jsonl", BaselineRunRecord)
    assert len(eval_records) == 1
    assert len(run_records) == 1
    assert run_records[0].reference_question.question_id == "q1"
    assert run_records[0].evaluation.human_review["quality_mean"] == 1.0
    summary = json.loads((output_dir / "test_baseline_summary.json").read_text(encoding="utf-8"))
    assert summary["total_runs"] == 1
    assert summary["rejection_rate"] == 0.0

    records_second = run_baseline_eval(
        split_path=split_path,
        split_name="test",
        output_dir=output_dir,
        questions_path=questions_path,
        reviews_path=reviews_path,
    )
    assert len(records_second) == 1
    assert len(read_jsonl(output_dir / "test_baseline_runs.jsonl", BaselineRunRecord)) == 1

    shutil.rmtree(tmp_path)


def test_generated_question_review_store_loads_generated_run_records() -> None:
    tmp_path = Path("tests_runtime") / make_id("generated_run_review")
    tmp_path.mkdir(parents=True, exist_ok=True)
    runs_path = tmp_path / "runs.jsonl"
    reviews_path = tmp_path / "reviews.jsonl"

    base = _build_run_record("draft_generated")
    generated = GeneratedQuestionRunRecord(
        run_id="generation_batch",
        job_id="bio_job",
        spec=base.spec,
        bundle=RetrievalBundle(spec_id=base.spec.spec_id),
        draft=base.draft,
        report=base.report,
        evaluation=base.evaluation,
        metadata={"job_id": "bio_job"},
    )
    runs_path.write_text(generated.model_dump_json() + "\n", encoding="utf-8")

    store = GeneratedQuestionReviewStore(runs_path, reviews_path, reviewer_id="brian")
    payload = store.question_payload("draft_generated")

    assert payload["metadata"]["job_id"] == "bio_job"
    assert payload["metadata"]["source_label"] == "bio_job"

    shutil.rmtree(tmp_path)


def _build_run_record(draft_id: str, *, candidate_question_id: str = "q_ref") -> BaselineRunRecord:
    spec = QuestionSpec(
        spec_id="spec_1",
        category=Category.BIOLOGY,
        subcategory="cell biology",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
        topic_focus=["mitochondria"],
    )
    draft = GeneratedDraft(
        draft_id=draft_id,
        spec_id=spec.spec_id,
        model_info=ModelInfo(provider="local", model_name="test-writer", prompt_version="writer_v1"),
        question=DraftQuestion(
            category=Category.BIOLOGY,
            subcategory="cell biology",
            question_type=QuestionType.TOSSUP,
            answer_mode=AnswerMode.SHORT_ANSWER,
            difficulty=4,
            question_text="Identify this organelle with its own DNA.",
            answer_text="ANSWER: mitochondrion",
        ),
    )
    report = VerifierReport(
        report_id="verify_1",
        draft_id=draft_id,
        spec_id=spec.spec_id,
        verdict=Verdict.PASS,
        summary="Looks acceptable.",
        checks=VerificationChecks(
            format_compliance=VerificationCheck(passed=True),
            factual_grounding=VerificationCheck(passed=True),
            answerability=VerificationCheck(passed=True),
            difficulty_alignment=VerificationCheck(passed=True, estimated_difficulty=4),
            topic_alignment=VerificationCheck(passed=True, style_score=0.6),
            style_alignment=VerificationCheck(passed=True, style_score=0.7),
            novelty=VerificationCheck(passed=True, similarity_score=0.2),
        ),
        review_metadata=ReviewMetadata(provider="local", model_name="test-verifier", prompt_version="verifier_v1"),
    )
    evaluation = EvaluationRecord(
        run_id="baseline_eval",
        candidate_question_id=candidate_question_id,
        spec_id=spec.spec_id,
        scores={"overall": 0.5},
    )
    reference_question = NormalizedQuestion(
        question_id=candidate_question_id,
        source_type=SourceType.DATASET,
        source_id="mit_2025",
        category=Category.BIOLOGY,
        subcategory="cell biology",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
        question_text="What organelle is the powerhouse of the cell?",
        answer_text="ANSWER: mitochondrion",
        source_metadata=SourceMetadata(round=3, tournament="MIT 2025"),
    )
    return BaselineRunRecord(
        run_id="baseline_eval",
        candidate_question_id=candidate_question_id,
        spec=spec,
        draft=draft,
        report=report,
        evaluation=evaluation,
        reference_question=reference_question,
        reference_human_review={"quality_mean": 1.0, "difficulty_mean": 4.0, "reviewed": True},
    )
