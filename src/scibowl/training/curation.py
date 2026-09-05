from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from scibowl.schema.common import AnswerMode, Category, SourceType
from scibowl.schema.question import NormalizedQuestion
from scibowl.schema.review import HumanReview
from scibowl.schema.training import ChatMessage, CleanSFTExample, CleanSFTMetadata
from scibowl.utils.ids import slugify
from scibowl.utils.io import read_jsonl, write_json, write_jsonl
from scibowl.utils.subcategories import canonicalize_subcategory
from scibowl.utils.text import normalize_whitespace


CLEAN_SFT_PROFILES = (
    'mit_recent',
    'mit_all',
    'mit_recent_plus_nsb',
    'mit_all_plus_nsb',
)

_UNKNOWN_SUBCATEGORIES = {'', 'other', 'unknown', 'unspecified', 'none'}
_NEAR_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'how',
    'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'these', 'this',
    'to', 'what', 'when', 'which', 'who', 'why', 'with', 'following',
}


@dataclass
class _Candidate:
    question: NormalizedQuestion
    source_family: str
    source_year: int | None
    subcategory: str | None
    subcategory_source: str | None
    difficulty: int | None
    difficulty_mean: float | None
    difficulty_source: str | None
    quality_mean: float | None
    quality_source: str | None
    exact_fingerprint: str
    near_fingerprint: str
    duplicate_source_question_ids: list[str] = field(default_factory=list)


def build_clean_sft_dataset(
    questions_path: Path | tuple[Path, ...],
    output_dir: Path,
    *,
    review_paths: tuple[Path, ...] = (),
    profile: str = 'mit_recent_plus_nsb',
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
    seed: int = 42,
    held_out_question_ids_path: Path | None = None,
    minimum_writing_quality: float = 0.0,
    include_unrated_writing: bool = False,
) -> dict[str, object]:
    if profile not in CLEAN_SFT_PROFILES:
        raise ValueError(f'Unknown clean SFT profile: {profile}')
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError('validation_fraction and test_fraction must be nonnegative and sum to less than 1')

    question_paths = (
        questions_path if isinstance(questions_path, tuple) else (questions_path,)
    )
    questions, duplicate_input_ids = _load_questions(question_paths)
    rating_map = _load_rating_map(review_paths)
    held_out_ids = _load_question_ids(held_out_question_ids_path)
    excluded_counts = {
        'profile': 0,
        'legacy_energy': 0,
        'missing_text': 0,
        'invalid_answer_line': 0,
        'contaminated_parse': 0,
        'malformed_multiple_choice': 0,
        'malformed_short_answer': 0,
        'below_minimum_writing_quality': 0,
        'unrated_writing': 0,
    }

    candidates: list[_Candidate] = []
    for question in questions:
        family = _source_family(question)
        year = _source_year(question)
        if not _profile_includes(profile, family, year):
            excluded_counts['profile'] += 1
            continue
        if question.category == Category.ENERGY and not (family == 'mit' and year is not None and year >= 2024):
            excluded_counts['legacy_energy'] += 1
            continue
        gate_reason = _gate_reason(question)
        if gate_reason is not None:
            excluded_counts[gate_reason] += 1
            continue
        candidate = _prepare_candidate(
            question,
            family,
            year,
            rating_map.get(question.question_id, []),
        )
        if _is_writing_source(question, family, year):
            if candidate.quality_mean is None and not include_unrated_writing:
                excluded_counts['unrated_writing'] += 1
                continue
            if (
                candidate.quality_mean is not None
                and candidate.quality_mean < minimum_writing_quality
            ):
                excluded_counts['below_minimum_writing_quality'] += 1
                continue
        candidates.append(candidate)

    unique_candidates, exact_duplicate_count = _dedupe_exact(candidates)
    input_question_ids = {question.question_id for question in questions}
    eligible_source_ids = {
        source_id
        for candidate in unique_candidates
        for source_id in (
            candidate.question.question_id,
            *candidate.duplicate_source_question_ids,
        )
    }
    held_out_ids_present = held_out_ids & input_question_ids
    held_out_ids_eligible = held_out_ids & eligible_source_ids
    held_out_groups = {
        candidate.near_fingerprint
        for candidate in unique_candidates
        if held_out_ids.intersection(
            {candidate.question.question_id, *candidate.duplicate_source_question_ids}
        )
    }

    examples: list[CleanSFTExample] = []
    split_counts = {'train': 0, 'val': 0, 'test': 0, 'held_out': 0}
    category_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    dataset_counts: dict[str, int] = {}
    year_counts: dict[str, int] = {}
    labeled_difficulty_count = 0
    labeled_subcategory_count = 0
    labeled_quality_count = 0

    for candidate in sorted(unique_candidates, key=lambda item: item.question.question_id):
        if candidate.near_fingerprint in held_out_groups:
            split = 'held_out'
        else:
            split = _assign_group_split(
                candidate.near_fingerprint,
                seed=seed,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
            )
        example = _build_example(candidate, profile=profile, split=split)
        examples.append(example)
        split_counts[split] += 1
        category = candidate.question.category.value
        category_counts[category] = category_counts.get(category, 0) + 1
        family_counts[candidate.source_family] = family_counts.get(candidate.source_family, 0) + 1
        dataset = candidate.question.source_id
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        year = str(candidate.source_year) if candidate.source_year is not None else 'unknown'
        year_counts[year] = year_counts.get(year, 0) + 1
        labeled_difficulty_count += int(candidate.difficulty is not None)
        labeled_subcategory_count += int(candidate.subcategory is not None)
        labeled_quality_count += int(candidate.quality_mean is not None)

    (output_dir / 'clean_sft_all.jsonl').unlink(missing_ok=True)
    for split in split_counts:
        write_jsonl(
            output_dir / f'clean_sft_{split}.jsonl',
            [example for example in examples if example.metadata.split == split],
        )

    summary = {
        'schema_version': 'clean_sft_v2',
        'questions_path': str(question_paths[0]) if len(question_paths) == 1 else None,
        'question_paths': [str(path) for path in question_paths],
        'review_paths': [str(path) for path in review_paths],
        'profile': profile,
        'input_questions': len(questions),
        'duplicate_input_question_ids_skipped': duplicate_input_ids,
        'eligible_before_dedupe': len(candidates),
        'unique_questions': len(unique_candidates),
        'exact_duplicates_removed': exact_duplicate_count,
        'excluded_counts': excluded_counts,
        'split_counts': split_counts,
        'category_counts': category_counts,
        'source_family_counts': family_counts,
        'source_dataset_counts': dataset_counts,
        'source_year_counts': year_counts,
        'label_counts': {
            'difficulty': labeled_difficulty_count,
            'subcategory': labeled_subcategory_count,
            'quality': labeled_quality_count,
        },
        'validation_fraction': validation_fraction,
        'test_fraction': test_fraction,
        'seed': seed,
        'held_out_question_ids_path': str(held_out_question_ids_path) if held_out_question_ids_path else None,
        'held_out_requested_count': len(held_out_ids),
        'held_out_present_in_input_count': len(held_out_ids_present),
        'held_out_eligible_count': len(held_out_ids_eligible),
        'held_out_excluded_question_ids': sorted(
            held_out_ids_present - held_out_ids_eligible
        ),
        'minimum_writing_quality': minimum_writing_quality,
        'include_unrated_writing': include_unrated_writing,
        'rules': {
            'unlabeled_difficulty_is_omitted': True,
            'unlabeled_subcategory_is_omitted': True,
            'sample_weight': 1.0,
            'legacy_energy_excluded': True,
            'duplicate_group_split': True,
            'unusable_writing_questions_excluded': True,
            'unrated_writing_questions_excluded': not include_unrated_writing,
        },
    }
    write_json(output_dir / 'clean_sft_manifest.json', summary)
    return summary


def _load_questions(paths: tuple[Path, ...]) -> tuple[list[NormalizedQuestion], int]:
    questions: list[NormalizedQuestion] = []
    by_id: dict[str, NormalizedQuestion] = {}
    duplicate_ids = 0
    for path in paths:
        for question in read_jsonl(path, NormalizedQuestion):
            incumbent = by_id.get(question.question_id)
            if incumbent is None:
                by_id[question.question_id] = question
                questions.append(question)
                continue
            if _exact_fingerprint(incumbent) != _exact_fingerprint(question):
                raise ValueError(
                    f'Conflicting records share question_id {question.question_id!r}'
                )
            duplicate_ids += 1
    return questions, duplicate_ids


def _load_rating_map(review_paths: tuple[Path, ...]) -> dict[str, list[HumanReview]]:
    ratings: dict[str, list[HumanReview]] = {}
    for path in review_paths:
        for review in read_jsonl(path, HumanReview):
            ratings.setdefault(review.question_id, []).append(review)
    return ratings


def _load_question_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    text = path.read_text(encoding='utf-8').strip()
    if not text:
        return set()
    if text.startswith('[') or text.startswith('{'):
        payload = json.loads(text)
        values = payload.get('question_ids', []) if isinstance(payload, dict) else payload
        return {str(value).strip() for value in values if str(value).strip()}
    return {line.strip() for line in text.splitlines() if line.strip()}


def _prepare_candidate(
    question: NormalizedQuestion,
    family: str,
    year: int | None,
    reviews: list[HumanReview],
) -> _Candidate:
    reviewed_difficulties = [
        review.ratings.difficulty
        for review in reviews
        if review.ratings.difficulty is not None and 1 <= review.ratings.difficulty <= 7
    ]
    reviewed_qualities = [
        review.ratings.quality
        for review in reviews
        if review.ratings.quality is not None
    ]

    difficulty_mean: float | None = None
    difficulty: int | None = None
    difficulty_source: str | None = None
    if reviewed_difficulties:
        difficulty_mean = sum(reviewed_difficulties) / len(reviewed_difficulties)
        difficulty = max(1, min(7, math.floor(difficulty_mean + 0.5)))
        difficulty_source = 'human_review'
    elif question.source_metadata.original_difficulty is not None:
        raw_difficulty = question.source_metadata.original_difficulty
        if 1 <= raw_difficulty <= 7:
            difficulty_mean = float(raw_difficulty)
            difficulty = max(1, min(7, math.floor(raw_difficulty + 0.5)))
            difficulty_source = 'source_metadata'

    quality_mean: float | None = None
    quality_source: str | None = None
    if reviewed_qualities:
        quality_mean = sum(reviewed_qualities) / len(reviewed_qualities)
        quality_source = 'human_review'
    elif question.source_metadata.original_quality is not None:
        quality_mean = float(question.source_metadata.original_quality)
        quality_source = 'source_metadata'

    raw_subcategory = canonicalize_subcategory(question.subcategory).strip()
    if raw_subcategory.casefold() in _UNKNOWN_SUBCATEGORIES:
        subcategory = None
        subcategory_source = None
    else:
        subcategory = raw_subcategory
        subcategory_source = 'source'

    return _Candidate(
        question=question,
        source_family=family,
        source_year=year,
        subcategory=subcategory,
        subcategory_source=subcategory_source,
        difficulty=difficulty,
        difficulty_mean=difficulty_mean,
        difficulty_source=difficulty_source,
        quality_mean=quality_mean,
        quality_source=quality_source,
        exact_fingerprint=_exact_fingerprint(question),
        near_fingerprint=_near_fingerprint(question),
    )


def _source_family(question: NormalizedQuestion) -> str:
    source = question.source_id.casefold()
    tournament = (question.source_metadata.tournament or '').casefold()
    if source.startswith('mit_') or 'mit science bowl' in tournament:
        return 'mit'
    if source.startswith('nsb_') or 'national science bowl' in tournament:
        return 'official_nsb'
    return 'other'


def _source_year(question: NormalizedQuestion) -> int | None:
    if question.source_metadata.year is not None:
        return question.source_metadata.year
    for value in (question.source_id, question.source_metadata.tournament or '', question.provenance.raw_file or ''):
        match = re.search(r'(?<!\d)(20\d{2})(?!\d)', value)
        if match:
            return int(match.group(1))
    return None


def _is_writing_source(
    question: NormalizedQuestion,
    family: str,
    year: int | None,
) -> bool:
    if 'writing_sheet' in question.style_tags:
        return True
    return (
        family == 'mit'
        and question.source_type == SourceType.DATASET
        and year is not None
        and year >= 2024
    )


def _profile_includes(profile: str, family: str, year: int | None) -> bool:
    recent_mit = family == 'mit' and year is not None and year >= 2024
    return {
        'mit_recent': recent_mit,
        'mit_all': family == 'mit',
        'mit_recent_plus_nsb': recent_mit or family == 'official_nsb',
        'mit_all_plus_nsb': family in {'mit', 'official_nsb'},
    }[profile]


def _gate_reason(question: NormalizedQuestion) -> str | None:
    question_text = normalize_whitespace(question.question_text)
    answer_text = normalize_whitespace(question.answer_text)
    if not question_text or not answer_text:
        return 'missing_text'
    if not answer_text.startswith('ANSWER:'):
        return 'invalid_answer_line'
    if 'ANSWER:' in question_text or 'ANSWER:' in answer_text.removeprefix('ANSWER:'):
        return 'contaminated_parse'
    if question.answer_mode == AnswerMode.MULTIPLE_CHOICE:
        if len(question.choices) != 4 or [choice.label for choice in question.choices] != ['W', 'X', 'Y', 'Z']:
            return 'malformed_multiple_choice'
    elif question.choices:
        return 'malformed_short_answer'
    return None


def _exact_fingerprint(question: NormalizedQuestion) -> str:
    choices = '|'.join(
        f'{choice.label.casefold()}:{normalize_whitespace(choice.text).casefold()}'
        for choice in question.choices
    )
    payload = '||'.join(
        (
            normalize_whitespace(question.question_text).casefold(),
            normalize_whitespace(question.answer_text).casefold(),
            choices,
        )
    )
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()


def _near_fingerprint(question: NormalizedQuestion) -> str:
    question_tokens = [
        token
        for token in re.findall(r'[a-z0-9]+', question.question_text.casefold())
        if token not in _NEAR_STOPWORDS
    ]
    answer_tokens = re.findall(
        r'[a-z0-9]+',
        question.answer_text.removeprefix('ANSWER:').casefold(),
    )
    payload = '|'.join(
        (
            question.category.value,
            question.question_type.value,
            ' '.join(sorted(question_tokens)),
            ' '.join(sorted(answer_tokens)),
        )
    )
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()


def _dedupe_exact(candidates: list[_Candidate]) -> tuple[list[_Candidate], int]:
    best_by_fingerprint: dict[str, _Candidate] = {}
    for candidate in candidates:
        incumbent = best_by_fingerprint.get(candidate.exact_fingerprint)
        if incumbent is None:
            best_by_fingerprint[candidate.exact_fingerprint] = candidate
            continue
        if _candidate_priority(candidate) > _candidate_priority(incumbent):
            candidate.duplicate_source_question_ids.extend(
                [incumbent.question.question_id, *incumbent.duplicate_source_question_ids]
            )
            best_by_fingerprint[candidate.exact_fingerprint] = candidate
        else:
            incumbent.duplicate_source_question_ids.extend(
                [candidate.question.question_id, *candidate.duplicate_source_question_ids]
            )
    unique = list(best_by_fingerprint.values())
    for candidate in unique:
        candidate.duplicate_source_question_ids = sorted(set(candidate.duplicate_source_question_ids))
    return unique, len(candidates) - len(unique)


def _candidate_priority(candidate: _Candidate) -> tuple[int, int, int, int, int, str]:
    recent_mit = candidate.source_family == 'mit' and (candidate.source_year or 0) >= 2024
    return (
        int(candidate.quality_mean is not None),
        int(candidate.difficulty is not None),
        int(recent_mit),
        int(candidate.source_family == 'mit'),
        int(candidate.subcategory is not None),
        candidate.question.question_id,
    )


def _assign_group_split(
    group: str,
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> str:
    digest = hashlib.sha256(f'{seed}:{group}'.encode('utf-8')).digest()
    score = int.from_bytes(digest[:8], 'big') / float(1 << 64)
    if score < test_fraction:
        return 'test'
    if score < test_fraction + validation_fraction:
        return 'val'
    return 'train'


def _build_example(candidate: _Candidate, *, profile: str, split: str) -> CleanSFTExample:
    question = candidate.question
    user_lines = [
        f'Category: {question.category.value}',
        f'Type: {question.question_type.value}',
    ]
    if candidate.subcategory is not None:
        user_lines.append(f'Subcategory: {candidate.subcategory}')
    if candidate.difficulty is not None:
        user_lines.append(f'Difficulty: {candidate.difficulty}')
    if question.content_tags:
        topic_focus = ', '.join(question.content_tags)
        user_lines.append(f'Topic focus: {topic_focus}')
    if question.category == Category.ENERGY:
        user_lines.append(
            'Energy format: begin with real current MIT research, then ask a related science or engineering question.'
        )
    user_lines.append('Write one new Science Bowl question. Return only JSON.')

    assistant_content = json.dumps(
        {
            'question_text': question.question_text,
            'answer_text': question.answer_text,
            'choices': [choice.model_dump() for choice in question.choices],
        },
        ensure_ascii=False,
        separators=(',', ':'),
    )
    return CleanSFTExample(
        example_id=f'clean_sft_{slugify(question.question_id)}',
        messages=[
            ChatMessage(role='system', content=_system_prompt()),
            ChatMessage(role='user', content='\n'.join(user_lines)),
            ChatMessage(role='assistant', content=assistant_content),
        ],
        metadata=CleanSFTMetadata(
            source_question_id=question.question_id,
            duplicate_source_question_ids=candidate.duplicate_source_question_ids,
            source_dataset=question.source_id,
            source_family=candidate.source_family,
            source_year=candidate.source_year,
            category=question.category,
            subcategory=candidate.subcategory,
            subcategory_source=candidate.subcategory_source,
            question_type=question.question_type,
            target_answer_mode=question.answer_mode,
            difficulty=candidate.difficulty,
            difficulty_mean=candidate.difficulty_mean,
            difficulty_source=candidate.difficulty_source,
            quality_mean=candidate.quality_mean,
            quality_source=candidate.quality_source,
            profile=profile,
            split=split,
            duplicate_group=candidate.near_fingerprint,
            exact_fingerprint=candidate.exact_fingerprint,
            near_fingerprint=candidate.near_fingerprint,
        ),
    )


def _system_prompt() -> str:
    return '\n'.join(
        (
            'You write questions for MIT Science Bowl.',
            'Match recent MIT style while keeping the wording original.',
            'For tossups, put the key clue or target early so speed and prediction matter.',
            'Bonuses may require more reasoning or computation.',
            'Use short answer or four choices labeled W, X, Y, Z as the content requires.',
            'Return only valid JSON with question_text, answer_text, and choices.',
        )
    )
