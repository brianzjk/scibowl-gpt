import shutil
from pathlib import Path

from scibowl.schema.common import AnswerMode, Category, QuestionType, SourceType
from scibowl.schema.question import NormalizedQuestion, SourceMetadata
from scibowl.schema.review import HumanReview, ReviewRatings
from scibowl.schema.training import CleanSFTExample
from scibowl.training.curation import build_clean_sft_dataset
from scibowl.utils.ids import make_id
from scibowl.utils.io import read_jsonl, write_jsonl


def _make_temp_dir() -> Path:
    path = Path('tests_runtime') / make_id('clean_sft')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _question(
    question_id: str,
    source_id: str,
    category: Category,
    question_text: str,
    answer_text: str,
    *,
    year: int | None = None,
    subcategory: str = 'other',
    difficulty: int = 3,
    quality: float | None = None,
) -> NormalizedQuestion:
    return NormalizedQuestion(
        question_id=question_id,
        source_type=(SourceType.DATASET if source_id.startswith('mit_') else SourceType.PACKET),
        source_id=source_id,
        category=category,
        subcategory=subcategory,
        question_type=QuestionType.TOSSUP,
        answer_mode=AnswerMode.SHORT_ANSWER,
        difficulty=difficulty,
        question_text=question_text,
        answer_text=answer_text,
        source_metadata=SourceMetadata(year=year, original_quality=quality),
    )


def test_clean_curation_uses_human_labels_and_keeps_nsb_labels_unknown() -> None:
    tmp_path = _make_temp_dir()
    questions_path = tmp_path / 'questions.jsonl'
    reviews_path = tmp_path / 'reviews.jsonl'
    output_dir = tmp_path / 'out'
    questions = [
        _question(
            'mit_2025_1',
            'mit_2025',
            Category.BIOLOGY,
            'What organelle carries out oxidative phosphorylation?',
            'ANSWER: mitochondrion',
            year=2025,
            subcategory='Cell Biology',
        ),
        _question(
            'nsb_1',
            'nsb_set_17',
            Category.PHYSICS,
            'What conserved quantity follows from time translation symmetry?',
            'ANSWER: energy',
        ),
        _question(
            'nsb_energy',
            'nsb_set_17',
            Category.ENERGY,
            'What fuel is refined from crude oil?',
            'ANSWER: gasoline',
        ),
        _question(
            'mit_energy_2025',
            'mit_2025',
            Category.ENERGY,
            'MIT researchers trained a language model to predict protein structure. What class of molecule is a protein?',
            'ANSWER: polymer',
            year=2025,
            subcategory='Artificial Intelligence',
            quality=0,
        ),
        _question(
            'mit_2023_1',
            'mit_2023',
            Category.CHEMISTRY,
            'What is the conjugate base of water?',
            'ANSWER: hydroxide',
            year=2023,
        ),
    ]
    reviews = [
        HumanReview(
            review_id='review_1',
            question_id='mit_2025_1',
            reviewer_id='reviewer_a',
            ratings=ReviewRatings(difficulty=4, quality=0.5),
        ),
        HumanReview(
            review_id='review_2',
            question_id='mit_2025_1',
            reviewer_id='reviewer_b',
            ratings=ReviewRatings(difficulty=5, quality=1.0),
        ),
    ]
    write_jsonl(questions_path, questions)
    write_jsonl(reviews_path, reviews)

    summary = build_clean_sft_dataset(
        questions_path,
        output_dir,
        review_paths=(reviews_path,),
        validation_fraction=0,
        test_fraction=0,
    )
    examples = read_jsonl(output_dir / 'clean_sft_all.jsonl', CleanSFTExample)
    by_id = {example.metadata.source_question_id: example for example in examples}

    mit = by_id['mit_2025_1']
    assert mit.metadata.difficulty == 5
    assert mit.metadata.difficulty_mean == 4.5
    assert mit.metadata.difficulty_source == 'human_review'
    assert mit.metadata.quality_mean == 0.75
    assert mit.metadata.subcategory == 'Cell Biology'
    assert 'Difficulty: 5' in mit.messages[1].content

    nsb = by_id['nsb_1']
    assert nsb.metadata.difficulty is None
    assert nsb.metadata.subcategory is None
    assert 'Difficulty:' not in nsb.messages[1].content
    assert 'Subcategory:' not in nsb.messages[1].content
    assert nsb.metadata.sample_weight == 1.0

    assert 'nsb_energy' not in by_id
    assert 'mit_2023_1' not in by_id
    assert 'Energy format:' in by_id['mit_energy_2025'].messages[1].content
    assert summary['excluded_counts']['legacy_energy'] == 1
    assert summary['excluded_counts']['profile'] == 1
    shutil.rmtree(tmp_path)


def test_clean_curation_quarantines_a_whole_near_duplicate_group() -> None:
    tmp_path = _make_temp_dir()
    questions_path = tmp_path / 'questions.jsonl'
    held_out_path = tmp_path / 'held_out.txt'
    output_dir = tmp_path / 'out'
    questions = [
        _question(
            'mit_a',
            'mit_2025',
            Category.PHYSICS,
            'What particle has negative charge?',
            'ANSWER: electron',
            year=2025,
            quality=0,
        ),
        _question(
            'mit_b',
            'mit_2025',
            Category.PHYSICS,
            'Which particle has a negative charge?',
            'ANSWER: electron',
            year=2025,
            quality=0,
        ),
    ]
    write_jsonl(questions_path, questions)
    held_out_path.write_text('mit_a\n', encoding='utf-8')

    summary = build_clean_sft_dataset(
        questions_path,
        output_dir,
        profile='mit_recent',
        validation_fraction=0.2,
        test_fraction=0.2,
        held_out_question_ids_path=held_out_path,
    )
    examples = read_jsonl(output_dir / 'clean_sft_all.jsonl', CleanSFTExample)

    assert len(examples) == 2
    assert {example.metadata.split for example in examples} == {'held_out'}
    assert len({example.metadata.duplicate_group for example in examples}) == 1
    assert summary['split_counts']['held_out'] == 2
    assert summary['split_counts']['train'] == 0
    assert summary['held_out_requested_count'] == 1
    assert summary['held_out_eligible_count'] == 1
    shutil.rmtree(tmp_path)


def test_clean_curation_exact_dedupe_prefers_recent_mit() -> None:
    tmp_path = _make_temp_dir()
    questions_path = tmp_path / 'questions.jsonl'
    output_dir = tmp_path / 'out'
    shared_text = 'What organelle contains chlorophyll?'
    shared_answer = 'ANSWER: chloroplast'
    questions = [
        _question('nsb_copy', 'nsb_set_17', Category.BIOLOGY, shared_text, shared_answer),
        _question(
            'mit_copy',
            'mit_2025',
            Category.BIOLOGY,
            shared_text,
            shared_answer,
            year=2025,
            subcategory='Cell Biology',
            quality=0,
        ),
    ]
    write_jsonl(questions_path, questions)

    summary = build_clean_sft_dataset(
        questions_path,
        output_dir,
        validation_fraction=0,
        test_fraction=0,
    )
    examples = read_jsonl(output_dir / 'clean_sft_train.jsonl', CleanSFTExample)

    assert len(examples) == 1
    assert examples[0].metadata.source_question_id == 'mit_copy'
    assert examples[0].metadata.duplicate_source_question_ids == ['nsb_copy']
    assert summary['exact_duplicates_removed'] == 1
    shutil.rmtree(tmp_path)


def test_clean_curation_excludes_unrated_and_rejected_writing_questions() -> None:
    tmp_path = _make_temp_dir()
    questions_path = tmp_path / 'questions.jsonl'
    output_dir = tmp_path / 'out'
    questions = [
        _question(
            'mit_usable',
            'mit_2026',
            Category.BIOLOGY,
            'What organelle contains the electron transport chain?',
            'ANSWER: mitochondrion',
            year=2026,
            quality=0,
        ),
        _question(
            'mit_rejected',
            'mit_2026',
            Category.BIOLOGY,
            'What organelle contains DNA?',
            'ANSWER: nucleus',
            year=2026,
            quality=-0.5,
        ),
        _question(
            'mit_unrated',
            'mit_2026',
            Category.BIOLOGY,
            'What organelle contains chlorophyll?',
            'ANSWER: chloroplast',
            year=2026,
        ),
    ]
    write_jsonl(questions_path, questions)

    summary = build_clean_sft_dataset(
        questions_path,
        output_dir,
        profile='mit_recent',
        validation_fraction=0,
        test_fraction=0,
    )
    examples = read_jsonl(output_dir / 'clean_sft_train.jsonl', CleanSFTExample)

    assert [example.metadata.source_question_id for example in examples] == ['mit_usable']
    assert summary['excluded_counts']['below_minimum_writing_quality'] == 1
    assert summary['excluded_counts']['unrated_writing'] == 1
    shutil.rmtree(tmp_path)


def test_clean_curation_reads_year_from_underscored_source_id() -> None:
    tmp_path = _make_temp_dir()
    questions_path = tmp_path / 'questions.jsonl'
    output_dir = tmp_path / 'out'
    question = _question(
        'mit_2024_official',
        'mit_2024',
        Category.CHEMISTRY,
        'What is the conjugate base of water?',
        'ANSWER: hydroxide',
    )
    question.source_type = SourceType.PACKET
    write_jsonl(questions_path, [question])

    summary = build_clean_sft_dataset(
        questions_path,
        output_dir,
        profile='mit_recent',
        validation_fraction=0,
        test_fraction=0,
    )

    assert summary['source_year_counts'] == {'2024': 1}
    assert summary['split_counts']['train'] == 1
    shutil.rmtree(tmp_path)


def test_clean_curation_combines_multiple_question_files() -> None:
    tmp_path = _make_temp_dir()
    first_path = tmp_path / 'first.jsonl'
    second_path = tmp_path / 'second.jsonl'
    output_dir = tmp_path / 'out'
    first = _question(
        'mit_first',
        'mit_2026',
        Category.MATH,
        'What is the derivative of x squared?',
        'ANSWER: two x',
        year=2026,
        quality=0,
    )
    second = _question(
        'mit_second',
        'mit_2026',
        Category.PHYSICS,
        'What force opposes relative motion?',
        'ANSWER: friction',
        year=2026,
        quality=0,
    )
    write_jsonl(first_path, [first])
    write_jsonl(second_path, [second])

    summary = build_clean_sft_dataset(
        (first_path, second_path),
        output_dir,
        profile='mit_recent',
        validation_fraction=0,
        test_fraction=0,
    )

    assert summary['input_questions'] == 2
    assert summary['question_paths'] == [str(first_path), str(second_path)]
    assert summary['split_counts']['train'] == 2
    shutil.rmtree(tmp_path)
