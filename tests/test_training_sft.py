import json
import shutil
from pathlib import Path

from scibowl.schema.common import AnswerMode, Category, QuestionType, SourceType
from scibowl.schema.question import Choice, NormalizedQuestion, Provenance, SourceMetadata
from scibowl.schema.training import SFTExample
from scibowl.training.sft import _materialized_repeat_count, export_sft_dataset
from scibowl.utils.ids import make_id
from scibowl.utils.io import read_json, read_jsonl


def _make_temp_dir() -> Path:
    path = Path("tests_runtime") / make_id("sft")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_questions(path: Path, questions: list[NormalizedQuestion]) -> None:
    path.write_text("\n".join(question.model_dump_json() for question in questions) + "\n", encoding="utf-8")


def test_export_sft_dataset_filters_weights_and_materializes() -> None:
    tmp_path = _make_temp_dir()
    questions_path = tmp_path / "questions.jsonl"
    output_dir = tmp_path / "out"

    mit_question = NormalizedQuestion(
        question_id="mit_2025_0001",
        source_type=SourceType.DATASET,
        source_id="mit_2025",
        category=Category.BIOLOGY,
        subcategory="Cell Biology",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.MULTIPLE_CHOICE,
        difficulty=3,
        question_text="Which of the following organelles contains its own DNA?",
        answer_text="ANSWER: X) mitochondrion",
        choices=[
            Choice(label="W", text="Golgi apparatus"),
            Choice(label="X", text="mitochondrion"),
            Choice(label="Y", text="lysosome"),
            Choice(label="Z", text="ribosome"),
        ],
        source_metadata=SourceMetadata(tournament="MIT Science Bowl 2025", year=2025, original_quality=1.0),
    )
    nsb_late_question = NormalizedQuestion(
        question_id="nsb_set_17__nsb_set_17_round_15_0007",
        source_type=SourceType.PACKET,
        source_id="nsb_set_17",
        category=Category.PHYSICS,
        subcategory="other",
        question_type=QuestionType.BONUS,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=4,
        question_text="What conserved quantity corresponds to invariance under time translation?",
        answer_text="ANSWER: energy",
        provenance=Provenance(raw_file="data\\raw\\packets\\nsb_set_17\\nsb_set_17__round_15.pdf"),
    )
    malformed_mc = NormalizedQuestion(
        question_id="packet_bad_0001",
        source_type=SourceType.PACKET,
        source_id="packet_bad",
        category=Category.CHEMISTRY,
        subcategory="other",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.MULTIPLE_CHOICE,
        difficulty=3,
        question_text="Select the correct acid.",
        answer_text="ANSWER: W) hydrochloric acid",
        choices=[
            Choice(label="W", text="hydrochloric acid"),
            Choice(label="X", text="sodium chloride"),
            Choice(label="Y", text="water"),
            Choice(label="Z", text="methane"),
        ],
    )
    contaminated_parse = NormalizedQuestion(
        question_id="packet_bad_0002",
        source_type=SourceType.PACKET,
        source_id="packet_bad",
        category=Category.CHEMISTRY,
        subcategory="other",
        question_type=QuestionType.BONUS,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=3,
        question_text="What statement about phase diagrams is true?",
        answer_text="ANSWER: None of them [EH] Toss-Up Amador Valley Science Bowl 2024 11) MATH Short Answer What is 2+2?",
    )
    energy_question = NormalizedQuestion(
        question_id="energy_0001",
        source_type=SourceType.PACKET,
        source_id="energy_set",
        category=Category.ENERGY,
        subcategory="other",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=3,
        question_text="What machine-learning model uses backpropagation?",
        answer_text="ANSWER: neural network",
    )

    _write_questions(
        questions_path,
        [mit_question, nsb_late_question, malformed_mc, contaminated_parse, energy_question],
    )

    summary = export_sft_dataset(questions_path, output_dir, validation_fraction=0.0, seed=42)

    assert summary["input_questions"] == 5
    assert summary["excluded_counts"]["excluded_category"] == 1
    assert summary["excluded_counts"]["malformed_multiple_choice"] == 1
    assert summary["excluded_counts"]["contaminated_parse"] == 1
    assert summary["unique_questions"] == 2
    assert summary["split_counts"]["train"] == 2
    assert summary["split_counts"]["val"] == 0

    train_examples = read_jsonl(output_dir / "sft_examples_train.jsonl", SFTExample)
    materialized_examples = read_jsonl(output_dir / "sft_examples_train_materialized.jsonl", SFTExample)
    summary_json = read_json(output_dir / "sft_summary.json")

    assert len(train_examples) == 2
    assert summary_json["weight_rules"]["mit_any_year"] == 2.0
    assert summary_json["weight_rules"]["official_nsb_round_gt_11"] == 1.5

    mit_example = next(example for example in train_examples if example.metadata.source_dataset == "mit_2025")
    nsb_example = next(example for example in train_examples if example.metadata.source_dataset == "nsb_set_17")

    assert mit_example.metadata.sample_weight == 2.0
    assert mit_example.metadata.weight_reasons == ["mit"]
    assert "Answer Mode:" not in mit_example.messages[1].content
    assert "Grounding Facts:" not in mit_example.messages[1].content
    assert "Style Examples:" not in mit_example.messages[1].content
    assert "Return only valid JSON" in mit_example.messages[0].content
    assert mit_example.metadata.target_answer_mode == AnswerMode.MULTIPLE_CHOICE

    assistant_payload = json.loads(mit_example.messages[2].content)
    assert assistant_payload["choices"][0]["label"] == "W"

    assert nsb_example.metadata.sample_weight == 1.5
    assert nsb_example.metadata.weight_reasons == ["late_official_nsb_round"]
    assert nsb_example.metadata.source_round_number == 15
    assert nsb_example.metadata.repeat_count == _materialized_repeat_count(
        nsb_example.metadata.source_question_id,
        nsb_example.metadata.sample_weight,
        seed=42,
    )

    assert len(materialized_examples) == mit_example.metadata.repeat_count + nsb_example.metadata.repeat_count
    shutil.rmtree(tmp_path)


def test_export_sft_dataset_dedupes_exact_questions_in_favor_of_mit() -> None:
    tmp_path = _make_temp_dir()
    questions_path = tmp_path / "questions.jsonl"
    output_dir = tmp_path / "out"

    duplicated_text = "Which of the following hydrocarbons contains the most carbon atoms per molecule?"
    duplicated_answer = "ANSWER: Z) Butane"
    duplicated_choices = [
        Choice(label="W", text="Methane"),
        Choice(label="X", text="Ethane"),
        Choice(label="Y", text="Propane"),
        Choice(label="Z", text="Butane"),
    ]

    packet_question = NormalizedQuestion(
        question_id="packet_0001",
        source_type=SourceType.PACKET,
        source_id="berkeley_2023",
        category=Category.CHEMISTRY,
        subcategory="other",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.MULTIPLE_CHOICE,
        difficulty=3,
        question_text=duplicated_text,
        answer_text=duplicated_answer,
        choices=duplicated_choices,
    )
    mit_question = NormalizedQuestion(
        question_id="mit_2024_0001",
        source_type=SourceType.DATASET,
        source_id="mit_2024",
        category=Category.CHEMISTRY,
        subcategory="Organic Chemistry",
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.MULTIPLE_CHOICE,
        difficulty=3,
        question_text=duplicated_text,
        answer_text=duplicated_answer,
        choices=duplicated_choices,
        source_metadata=SourceMetadata(tournament="MIT 2024", original_quality=1.0),
    )

    _write_questions(questions_path, [packet_question, mit_question])

    export_sft_dataset(questions_path, output_dir, validation_fraction=0.0, seed=7)
    train_examples = read_jsonl(output_dir / "sft_examples_train.jsonl", SFTExample)

    assert len(train_examples) == 1
    assert train_examples[0].metadata.source_dataset == "mit_2024"
    assert train_examples[0].metadata.sample_weight == 2.0
    shutil.rmtree(tmp_path)
