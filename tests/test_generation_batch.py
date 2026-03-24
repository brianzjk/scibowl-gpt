import json
import shutil
from pathlib import Path

import pytest

from scibowl.generate.batch import load_generation_batch_config, run_generation_batch
from scibowl.schema.common import AnswerMode, Category, Citation, ModelInfo, QuestionType, Verdict
from scibowl.schema.dataset import GeneratedQuestionRunRecord
from scibowl.schema.generation import DraftQuestion, GeneratedDraft, QuestionSpec, RetrievalBundle
from scibowl.schema.textbook import TextbookChunk
from scibowl.schema.verification import ReviewMetadata, VerificationCheck, VerificationChecks, VerifierReport
from scibowl.utils.ids import make_id
from scibowl.utils.io import read_jsonl


def test_load_generation_batch_config_rejects_energy() -> None:
    tmp_path = Path("tests_runtime") / make_id("generation_cfg")
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "output_path: output.jsonl",
                "style_questions_path: style.jsonl",
                "textbook_chunks_path: textbooks",
                "jobs:",
                "  - category: energy",
                "    subcategory: solar power",
                "    difficulty: 4",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_generation_batch_config(config_path)

    shutil.rmtree(tmp_path)


def test_run_generation_batch_writes_records(monkeypatch) -> None:
    tmp_path = Path("tests_runtime") / make_id("generation_batch")
    tmp_path.mkdir(parents=True, exist_ok=True)
    output_path = tmp_path / "generated.jsonl"
    style_path = tmp_path / "style.jsonl"
    textbooks_dir = tmp_path / "textbooks"
    textbooks_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yaml"

    style_path.write_text(
        json.dumps(
            {
                "question_id": "style_1",
                "source_type": "dataset",
                "source_id": "style",
                "category": "biology",
                "subcategory": "cell biology",
                "question_type": "tossup",
                "answer_mode": "short_answer",
                "difficulty": 4,
                "question_text": "What organelle contains DNA?",
                "answer_text": "ANSWER: mitochondrion",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (textbooks_dir / "campbell.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "chunk1",
                "document_id": "campbell_biology_12e",
                "source_type": "textbook",
                "title": "Campbell Biology",
                "pages": [],
                "topics": ["cells"],
                "text": "Mitochondria contain their own DNA.",
                "char_count": 34,
                "token_count_est": 6,
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        "\n".join(
            [
                f"output_path: {output_path.as_posix()}",
                f"style_questions_path: {style_path.as_posix()}",
                f"textbook_chunks_path: {textbooks_dir.as_posix()}",
                "jobs:",
                "  - job_id: bio_test",
                "    category: biology",
                "    subcategory: cell biology",
                "    question_type: tossup",
                "    answer_mode: short_answer",
                "    difficulty: 4",
                "    count: 1",
                "    topic_focus:",
                "      - mitochondria",
            ]
        ),
        encoding="utf-8",
    )

    class FakeOrchestrator:
        def run(self, spec, textbook_chunks, style_questions):
            draft = GeneratedDraft(
                draft_id="draft_1",
                spec_id=spec.spec_id,
                model_info=ModelInfo(provider="local", model_name="test-writer", prompt_version="writer_v1"),
                question=DraftQuestion(
                    category=Category.BIOLOGY,
                    subcategory="cell biology",
                    question_type=QuestionType.TOSSUP,
                    answer_mode=AnswerMode.SHORT_ANSWER,
                    difficulty=4,
                    question_text="What organelle contains its own DNA?",
                    answer_text="ANSWER: mitochondrion",
                ),
                citations=[Citation(source_id="campbell_biology_12e", chunk_id="chunk1")],
            )
            report = VerifierReport(
                report_id="verify_1",
                draft_id=draft.draft_id,
                spec_id=spec.spec_id,
                verdict=Verdict.PASS,
                summary="Looks usable.",
                checks=VerificationChecks(
                    format_compliance=VerificationCheck(passed=True),
                    factual_grounding=VerificationCheck(passed=True),
                    answerability=VerificationCheck(passed=True),
                    difficulty_alignment=VerificationCheck(passed=True, estimated_difficulty=4),
                    style_alignment=VerificationCheck(passed=True, style_score=0.8),
                    novelty=VerificationCheck(passed=True, similarity_score=0.1),
                ),
                review_metadata=ReviewMetadata(provider="local", model_name="test-verifier", prompt_version="verifier_v1"),
            )
            return RetrievalBundle(spec_id=spec.spec_id), draft, report

    monkeypatch.setattr("scibowl.generate.batch.GenerationOrchestrator", FakeOrchestrator)

    records = run_generation_batch(config_path)

    assert len(records) == 1
    written = read_jsonl(output_path, GeneratedQuestionRunRecord)
    assert len(written) == 1
    assert written[0].job_id == "bio_test"

    shutil.rmtree(tmp_path)


def test_run_generation_batch_supports_random_subcategory_mode(monkeypatch) -> None:
    tmp_path = Path("tests_runtime") / make_id("generation_batch_random")
    tmp_path.mkdir(parents=True, exist_ok=True)
    output_path = tmp_path / "generated.jsonl"
    style_path = tmp_path / "style.jsonl"
    textbooks_dir = tmp_path / "textbooks"
    textbooks_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yaml"

    style_path.write_text("", encoding="utf-8")
    config_path.write_text(
        "\n".join(
            [
                f"output_path: {output_path.as_posix()}",
                f"style_questions_path: {style_path.as_posix()}",
                f"textbook_chunks_path: {textbooks_dir.as_posix()}",
                "random_seed: 7",
                "jobs:",
                "  - job_id: math_random",
                "    category: math",
                "    subcategory_mode: random",
                "    question_type: tossup",
                "    answer_mode: short_answer",
                "    difficulty: 4",
                "    count: 3",
            ]
        ),
        encoding="utf-8",
    )

    class FakeOrchestrator:
        def run(self, spec, textbook_chunks, style_questions):
            draft = GeneratedDraft(
                draft_id=f"draft_{spec.subcategory}",
                spec_id=spec.spec_id,
                model_info=ModelInfo(provider="local", model_name="test-writer", prompt_version="writer_v1"),
                question=DraftQuestion(
                    category=spec.category,
                    subcategory=spec.subcategory,
                    question_type=spec.question_type,
                    answer_mode=spec.answer_mode,
                    difficulty=spec.difficulty,
                    question_text=f"Test question about {spec.subcategory}.",
                    answer_text="ANSWER: test",
                ),
            )
            report = VerifierReport(
                report_id=f"verify_{spec.subcategory}",
                draft_id=draft.draft_id,
                spec_id=spec.spec_id,
                verdict=Verdict.PASS,
                summary="Looks usable.",
                checks=VerificationChecks(
                    format_compliance=VerificationCheck(passed=True),
                    factual_grounding=VerificationCheck(passed=True),
                    answerability=VerificationCheck(passed=True),
                    difficulty_alignment=VerificationCheck(passed=True, estimated_difficulty=4),
                    style_alignment=VerificationCheck(passed=True, style_score=0.8),
                    novelty=VerificationCheck(passed=True, similarity_score=0.1),
                ),
                review_metadata=ReviewMetadata(provider="local", model_name="test-verifier", prompt_version="verifier_v1"),
            )
            return RetrievalBundle(spec_id=spec.spec_id), draft, report

    monkeypatch.setattr("scibowl.generate.batch.GenerationOrchestrator", FakeOrchestrator)

    records = run_generation_batch(config_path)

    assert len(records) == 3
    written = read_jsonl(output_path, GeneratedQuestionRunRecord)
    assert len(written) == 3
    assert {record.spec.subcategory for record in written}.issubset(
        {"Algebra", "Calculus", "Combinatorics", "Geometry", "Number Theory"}
    )
    assert all(record.metadata["subcategory_mode"] == "random" for record in written)

    shutil.rmtree(tmp_path)


def test_run_generation_batch_supports_random_question_shape_and_difficulty(monkeypatch) -> None:
    tmp_path = Path("tests_runtime") / make_id("generation_batch_random_shape")
    tmp_path.mkdir(parents=True, exist_ok=True)
    output_path = tmp_path / "generated.jsonl"
    style_path = tmp_path / "style.jsonl"
    textbooks_dir = tmp_path / "textbooks"
    textbooks_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yaml"

    style_path.write_text("", encoding="utf-8")
    config_path.write_text(
        "\n".join(
            [
                f"output_path: {output_path.as_posix()}",
                f"style_questions_path: {style_path.as_posix()}",
                f"textbook_chunks_path: {textbooks_dir.as_posix()}",
                "random_seed: 11",
                "jobs:",
                "  - category: earth_space",
                "    subcategory_mode: random",
                "    question_type_mode: random",
                "    answer_mode_mode: random",
                "    difficulty_mode: random",
                "    difficulty_pool: [2, 6]",
                "    count: 4",
            ]
        ),
        encoding="utf-8",
    )

    class FakeOrchestrator:
        def run(self, spec, textbook_chunks, style_questions):
            draft = GeneratedDraft(
                draft_id=f"draft_{spec.subcategory}_{spec.question_type.value}_{spec.answer_mode.value}_{spec.difficulty}",
                spec_id=spec.spec_id,
                model_info=ModelInfo(provider="local", model_name="test-writer", prompt_version="writer_v1"),
                question=DraftQuestion(
                    category=spec.category,
                    subcategory=spec.subcategory,
                    question_type=spec.question_type,
                    answer_mode=spec.answer_mode,
                    difficulty=spec.difficulty,
                    question_text=f"Test question about {spec.subcategory}.",
                    answer_text="ANSWER: test",
                ),
            )
            report = VerifierReport(
                report_id=f"verify_{spec.subcategory}",
                draft_id=draft.draft_id,
                spec_id=spec.spec_id,
                verdict=Verdict.PASS,
                summary="Looks usable.",
                checks=VerificationChecks(
                    format_compliance=VerificationCheck(passed=True),
                    factual_grounding=VerificationCheck(passed=True),
                    answerability=VerificationCheck(passed=True),
                    difficulty_alignment=VerificationCheck(passed=True, estimated_difficulty=spec.difficulty),
                    style_alignment=VerificationCheck(passed=True, style_score=0.8),
                    novelty=VerificationCheck(passed=True, similarity_score=0.1),
                ),
                review_metadata=ReviewMetadata(provider="local", model_name="test-verifier", prompt_version="verifier_v1"),
            )
            return RetrievalBundle(spec_id=spec.spec_id), draft, report

    monkeypatch.setattr("scibowl.generate.batch.GenerationOrchestrator", FakeOrchestrator)

    records = run_generation_batch(config_path)

    assert len(records) == 4
    written = read_jsonl(output_path, GeneratedQuestionRunRecord)
    assert len(written) == 4
    assert {record.spec.question_type for record in written}.issubset({QuestionType.TOSSUP, QuestionType.BONUS})
    assert {record.spec.answer_mode for record in written}.issubset(
        {AnswerMode.SHORT_ANSWER, AnswerMode.MULTIPLE_CHOICE}
    )
    assert {record.spec.difficulty for record in written}.issubset({2, 6})
    assert all(record.metadata["question_type_mode"] == "random" for record in written)
    assert all(record.metadata["answer_mode_mode"] == "random" for record in written)
    assert all(record.metadata["difficulty_mode"] == "random" for record in written)

    shutil.rmtree(tmp_path)
