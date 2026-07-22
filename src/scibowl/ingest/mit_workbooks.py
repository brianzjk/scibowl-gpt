from __future__ import annotations

import hashlib
import math
import posixpath
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

from scibowl.ingest.question_sets import _normalize_category, _split_guidance
from scibowl.schema.common import AnswerMode, QuestionType, SourceType
from scibowl.schema.question import (
    AnswerGuidance,
    Choice,
    NormalizedQuestion,
    Provenance,
    SourceMetadata,
)
from scibowl.schema.review import HumanReview, ReviewRatings, SourceComment
from scibowl.utils.ids import slugify
from scibowl.utils.text import normalize_whitespace


_REL_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
_PACKAGE_REL_NS = 'http://schemas.openxmlformats.org/package/2006/relationships'
_CELL_RE = re.compile(r'([A-Z]+)(\d+)')
_AVERAGE_RANGE_RE = re.compile(
    r'AVERAGE\(\$?([A-Z]+)\$?\d+:\$?([A-Z]+)\$?\d+\)',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MitWritingWorkbookImport:
    questions: list[NormalizedQuestion]
    reviews: list[HumanReview]
    comments: list[SourceComment]
    summary: dict[str, object]


@dataclass(frozen=True)
class _Cell:
    value: object | None
    formula: str | None


@dataclass(frozen=True)
class _RawComment:
    comment_id: str
    cell_ref: str
    author: str | None
    text: str
    timestamp: datetime | None
    parent_comment_id: str | None
    resolved: bool | None
    kind: str


@dataclass
class _Sheet:
    name: str
    cells: dict[str, _Cell]
    comments: list[_RawComment]

    def cell(self, row: int, column: int) -> _Cell:
        return self.cells.get(f'{_column_name(column)}{row}', _Cell(None, None))

    @property
    def max_row(self) -> int:
        return max((_split_cell_ref(ref)[1] for ref in self.cells), default=0)


def normalize_mit_writing_workbooks(
    input_dir: Path,
    *,
    source_id: str = 'mit_2026',
    tournament: str = 'MIT Science Bowl 2026',
    year: int = 2026,
) -> MitWritingWorkbookImport:
    paths = sorted(input_dir.glob('*.xlsx'))
    if not paths:
        raise ValueError(f'No .xlsx workbooks found in {input_dir}')

    questions: list[NormalizedQuestion] = []
    reviews: list[HumanReview] = []
    comments: list[SourceComment] = []
    file_summaries: list[dict[str, object]] = []

    for path in paths:
        imported = _normalize_workbook(
            path,
            source_id=source_id,
            tournament=tournament,
            year=year,
        )
        questions.extend(imported.questions)
        reviews.extend(imported.reviews)
        comments.extend(imported.comments)
        file_summaries.append(imported.summary)

    question_ids = [question.question_id for question in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError('Duplicate 2026 question IDs were produced')

    summary = {
        'schema_version': 'mit_writing_xlsx_v1',
        'source_id': source_id,
        'tournament': tournament,
        'year': year,
        'input_dir': str(input_dir),
        'source_files': [str(path) for path in paths],
        'file_count': len(paths),
        'question_count': len(questions),
        'review_count': len(reviews),
        'comment_count': len(comments),
        'files': file_summaries,
        'rules': {
            'visual_bonuses_excluded': True,
            'unrated_questions_preserved': True,
            'difficulty_range': [1, 7],
            'quality_range': [-1, 1],
            'invalid_ratings_ignored': True,
            'comments_are_sidecar_only': True,
        },
    }
    return MitWritingWorkbookImport(questions, reviews, comments, summary)


def _normalize_workbook(
    path: Path,
    *,
    source_id: str,
    tournament: str,
    year: int,
) -> MitWritingWorkbookImport:
    subject = _subject_from_path(path)
    with _WorkbookReader(path) as reader:
        sheet = reader.sheet('Template')
        modified_at = reader.modified_at()

    difficulty_columns = _rating_columns(sheet, average_column=14, prefix='D')
    quality_columns = _rating_columns(sheet, average_column=15, prefix='Q')
    reviewer_labels = _reviewer_labels(sheet, difficulty_columns | quality_columns)
    question_by_row: dict[int, NormalizedQuestion] = {}
    reviews: list[HumanReview] = []
    invalid_rating_count = 0
    skip_counts = {
        'blank': 0,
        'visual_bonus': 0,
        'missing_required': 0,
        'invalid_type': 0,
        'invalid_format': 0,
        'invalid_category': 0,
    }
    malformed_multiple_choice_count = 0

    for row_number in range(2, sheet.max_row + 1):
        row = {column: sheet.cell(row_number, column).value for column in range(1, 16)}
        if not any(value not in (None, '') for value in row.values()):
            skip_counts['blank'] += 1
            continue

        type_text = _text(row[1])
        if 'visual bonus' in type_text.casefold():
            skip_counts['visual_bonus'] += 1
            continue
        question_type = _strict_question_type(type_text)
        if question_type is None:
            skip_counts['invalid_type'] += 1
            continue

        answer_mode = _strict_answer_mode(_text(row[3]))
        if answer_mode is None:
            skip_counts['invalid_format'] += 1
            continue

        question_text = _text(row[4])
        answer_raw = _text(row[9])
        category_text = _text(row[2]) or subject
        if not question_text or not answer_raw:
            skip_counts['missing_required'] += 1
            continue
        try:
            category = _normalize_category(category_text)
        except ValueError:
            skip_counts['invalid_category'] += 1
            continue

        choices: list[Choice] = []
        if answer_mode == AnswerMode.MULTIPLE_CHOICE:
            for column, label in zip(range(5, 9), ('W', 'X', 'Y', 'Z'), strict=True):
                choice_text = _text(row[column])
                if choice_text:
                    choices.append(Choice(label=label, text=choice_text))
            if len(choices) != 4:
                malformed_multiple_choice_count += 1
            matched = next(
                (choice for choice in choices if choice.label == answer_raw.upper()),
                None,
            )
            answer_text = (
                f'ANSWER: {matched.label}) {matched.text}'
                if matched is not None
                else f'ANSWER: {answer_raw}'
            )
        else:
            answer_text = f'ANSWER: {answer_raw}'

        difficulty_values, bad_difficulty = _ratings_for_row(
            sheet,
            row_number,
            difficulty_columns,
            minimum=1,
            maximum=7,
        )
        quality_values, bad_quality = _ratings_for_row(
            sheet,
            row_number,
            quality_columns,
            minimum=-1,
            maximum=1,
        )
        invalid_rating_count += bad_difficulty + bad_quality
        difficulty_mean = _mean(value for _, value in difficulty_values)
        quality_mean = _mean(value for _, value in quality_values)
        if difficulty_mean is None:
            difficulty_mean = _valid_number(row[14], minimum=1, maximum=7)
        if quality_mean is None:
            quality_mean = _valid_number(row[15], minimum=-1, maximum=1)

        question_id = f'{source_id}_{slugify(subject)}_{row_number:04d}'
        subcategory = _text(row[12]) or 'other'
        writer = _text(row[13]) or None
        review_status = 'unrated'
        if quality_mean is not None:
            review_status = 'rated_usable' if quality_mean >= 0 else 'rated_rejected'
        question = NormalizedQuestion(
            question_id=question_id,
            source_type=SourceType.DATASET,
            source_id=source_id,
            category=category,
            subcategory=subcategory,
            question_type=question_type,
            answer_mode=answer_mode,
            difficulty=(
                max(1, min(7, math.floor(difficulty_mean + 0.5)))
                if difficulty_mean is not None
                else 3
            ),
            question_text=question_text,
            answer_text=answer_text,
            choices=choices,
            answer_guidance=AnswerGuidance(
                accept=_split_guidance(_text(row[10])),
                do_not_accept=_split_guidance(_text(row[11])),
            ),
            style_tags=['science_bowl', 'mit', 'writing_sheet'],
            content_tags=[subcategory] if subcategory != 'other' else [],
            source_metadata=SourceMetadata(
                source_row=row_number,
                tournament=tournament,
                year=year,
                original_difficulty=difficulty_mean,
                original_quality=quality_mean,
                writer=writer,
            ),
            provenance=Provenance(
                raw_file=str(path),
                parser_version='mit_writing_xlsx_v1',
                review_status=review_status,
            ),
        )
        question_by_row[row_number] = question
        reviews.extend(
            _reviews_for_ratings(
                question,
                path=path,
                sheet_name=sheet.name,
                subject=subject,
                tournament=tournament,
                modified_at=modified_at,
                kind='difficulty',
                ratings=difficulty_values,
                reviewer_labels=reviewer_labels,
            )
        )
        reviews.extend(
            _reviews_for_ratings(
                question,
                path=path,
                sheet_name=sheet.name,
                subject=subject,
                tournament=tournament,
                modified_at=modified_at,
                kind='quality',
                ratings=quality_values,
                reviewer_labels=reviewer_labels,
            )
        )

    normalized_comments = [
        SourceComment(
            comment_id=f'{source_id}_{slugify(subject)}_{comment.comment_id}',
            source_id=source_id,
            question_id=_question_id_for_comment(comment, question_by_row),
            raw_file=str(path),
            sheet_name=sheet.name,
            cell_ref=comment.cell_ref,
            source_row=_split_cell_ref(comment.cell_ref)[1],
            author=comment.author,
            text=comment.text,
            timestamp=comment.timestamp,
            parent_comment_id=(
                f'{source_id}_{slugify(subject)}_{comment.parent_comment_id}'
                if comment.parent_comment_id
                else None
            ),
            resolved=comment.resolved,
            comment_kind=comment.kind,
        )
        for comment in sheet.comments
        if comment.text and _valid_cell_ref(comment.cell_ref)
    ]

    questions = list(question_by_row.values())
    summary = {
        'raw_file': str(path),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'subject': subject,
        'template_max_row': sheet.max_row,
        'question_count': len(questions),
        'review_count': len(reviews),
        'comment_count': len(normalized_comments),
        'rated_difficulty_count': sum(
            question.source_metadata.original_difficulty is not None for question in questions
        ),
        'rated_quality_count': sum(
            question.source_metadata.original_quality is not None for question in questions
        ),
        'unrated_quality_count': sum(
            question.source_metadata.original_quality is None for question in questions
        ),
        'invalid_rating_count': invalid_rating_count,
        'malformed_multiple_choice_count': malformed_multiple_choice_count,
        'skip_counts': skip_counts,
        'difficulty_columns': [_column_name(column) for column in sorted(difficulty_columns)],
        'quality_columns': [_column_name(column) for column in sorted(quality_columns)],
    }
    return MitWritingWorkbookImport(questions, reviews, normalized_comments, summary)


def _reviews_for_ratings(
    question: NormalizedQuestion,
    *,
    path: Path,
    sheet_name: str,
    subject: str,
    tournament: str,
    modified_at: datetime,
    kind: str,
    ratings: list[tuple[int, float]],
    reviewer_labels: dict[int, list[str]],
) -> list[HumanReview]:
    output: list[HumanReview] = []
    source_row = question.source_metadata.source_row
    assert source_row is not None
    for column, value in ratings:
        labels = reviewer_labels.get(column, [])
        slot = _column_name(column)
        label = ' / '.join(labels) if labels else f'{subject} {kind} {slot}'
        reviewer_id = slugify(label) or f'{slugify(subject)}_{kind}_{slugify(slot)}'
        output.append(
            HumanReview(
                review_id=f'{question.question_id}_{kind}_{slot.casefold()}',
                question_id=question.question_id,
                reviewer_id=reviewer_id,
                tournament=tournament,
                review_timestamp=modified_at,
                ratings=ReviewRatings(
                    difficulty=value if kind == 'difficulty' else None,
                    quality=value if kind == 'quality' else None,
                ),
                source_row=source_row,
                metadata={
                    'raw_file': str(path),
                    'sheet_name': sheet_name,
                    'cell_ref': f'{slot}{source_row}',
                    'rating_kind': kind,
                    'rating_slot': slot,
                    'reviewer_labels': labels,
                },
            )
        )
    return output


def _rating_columns(sheet: _Sheet, *, average_column: int, prefix: str) -> set[int]:
    for row in range(2, min(sheet.max_row, 50) + 1):
        formula = sheet.cell(row, average_column).formula or ''
        match = _AVERAGE_RANGE_RE.search(formula)
        if match:
            start = _column_number(match.group(1))
            end = _column_number(match.group(2))
            return set(range(start, end + 1))

    return {
        column
        for column in range(16, 200)
        if re.fullmatch(
            fr'{prefix}\d+',
            _text(sheet.cell(1, column).value),
            re.IGNORECASE,
        )
    }


def _reviewer_labels(sheet: _Sheet, columns: set[int]) -> dict[int, list[str]]:
    output: dict[int, list[str]] = {}
    for column in columns:
        cell_ref = f'{_column_name(column)}1'
        labels: list[str] = []
        for comment in sheet.comments:
            if comment.cell_ref != cell_ref or comment.parent_comment_id is not None:
                continue
            label = normalize_whitespace(comment.text) or normalize_whitespace(
                comment.author or ''
            )
            if label and label not in labels:
                labels.append(label)
        output[column] = sorted(labels, key=str.casefold)
    return output


def _ratings_for_row(
    sheet: _Sheet,
    row: int,
    columns: set[int],
    *,
    minimum: float,
    maximum: float,
) -> tuple[list[tuple[int, float]], int]:
    output: list[tuple[int, float]] = []
    invalid = 0
    for column in sorted(columns):
        raw = sheet.cell(row, column).value
        if raw in (None, ''):
            continue
        value = _number(raw)
        if value is None or not minimum <= value <= maximum:
            invalid += 1
            continue
        output.append((column, value))
    return output, invalid


def _valid_number(value: object, *, minimum: float, maximum: float) -> float | None:
    number = _number(value)
    if number is None or not minimum <= number <= maximum:
        return None
    return number


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(normalize_whitespace(str(value)))
    except (TypeError, ValueError):
        return None


def _mean(values: object) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def _strict_question_type(value: str) -> QuestionType | None:
    normalized = slugify(value)
    if normalized in {'tossup', 'toss_up'}:
        return QuestionType.TOSSUP
    if normalized == 'bonus':
        return QuestionType.BONUS
    return None


def _strict_answer_mode(value: str) -> AnswerMode | None:
    normalized = slugify(value)
    if normalized in {'multiple_choice', 'multiplechoice'}:
        return AnswerMode.MULTIPLE_CHOICE
    if normalized in {'short_answer', 'shortanswer'}:
        return AnswerMode.SHORT_ANSWER
    return None


def _subject_from_path(path: Path) -> str:
    match = re.match(r'\d{4}\s+(.+?)\s+Question Writing', path.stem, re.IGNORECASE)
    if not match:
        raise ValueError(f'Cannot infer subject from workbook name: {path.name}')
    return normalize_whitespace(match.group(1))


def _text(value: object) -> str:
    return '' if value is None else normalize_whitespace(str(value))


def _valid_cell_ref(ref: str) -> bool:
    return _CELL_RE.fullmatch(ref) is not None


def _question_id_for_comment(
    comment: _RawComment,
    question_by_row: dict[int, NormalizedQuestion],
) -> str | None:
    if not _valid_cell_ref(comment.cell_ref):
        return None
    question = question_by_row.get(_split_cell_ref(comment.cell_ref)[1])
    return question.question_id if question is not None else None


def _split_cell_ref(ref: str) -> tuple[int, int]:
    match = _CELL_RE.fullmatch(ref)
    if not match:
        raise ValueError(f'Invalid Excel cell reference: {ref}')
    return _column_number(match.group(1)), int(match.group(2))


def _column_number(name: str) -> int:
    value = 0
    for character in name.upper():
        value = value * 26 + ord(character) - ord('A') + 1
    return value


def _column_name(number: int) -> str:
    output = ''
    while number:
        number, remainder = divmod(number - 1, 26)
        output = chr(ord('A') + remainder) + output
    return output


class _WorkbookReader:
    def __init__(self, path: Path):
        self.path = path
        self.archive = zipfile.ZipFile(path)
        self.names = set(self.archive.namelist())
        self.shared_strings = self._load_shared_strings()
        self.persons = self._load_persons()

    def __enter__(self) -> _WorkbookReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.archive.close()

    def sheet(self, name: str) -> _Sheet:
        workbook = ET.fromstring(self.archive.read('xl/workbook.xml'))
        relationships = self._relationships('xl/_rels/workbook.xml.rels')
        target: str | None = None
        for element in workbook.iter():
            if element.tag.endswith('}sheet') and element.get('name') == name:
                target = relationships[element.get(f'{{{_REL_NS}}}id')]
                break
        if target is None:
            raise ValueError(f'Workbook {self.path} has no {name!r} sheet')
        target = self._normalize_target('xl', target)
        root = ET.fromstring(self.archive.read(target))
        cells: dict[str, _Cell] = {}
        for element in root.iter():
            if not element.tag.endswith('}c'):
                continue
            ref = element.get('r')
            if ref:
                cells[ref] = self._decode_cell(element)
        return _Sheet(
            name=name,
            cells=cells,
            comments=self._load_sheet_comments(target),
        )

    def modified_at(self) -> datetime:
        if 'docProps/core.xml' in self.names:
            root = ET.fromstring(self.archive.read('docProps/core.xml'))
            for element in root.iter():
                if element.tag.endswith('}modified') and element.text:
                    parsed = _parse_datetime(element.text)
                    if parsed is not None:
                        return parsed
        return datetime.fromtimestamp(self.path.stat().st_mtime, tz=timezone.utc)

    def _decode_cell(self, element: ET.Element) -> _Cell:
        formula_element = next(
            (child for child in element if child.tag.endswith('}f')),
            None,
        )
        formula = formula_element.text if formula_element is not None else None
        cell_type = element.get('t')
        if cell_type == 'inlineStr':
            inline = next((child for child in element if child.tag.endswith('}is')), None)
            return _Cell(_xml_text(inline), formula)
        value_element = next(
            (child for child in element if child.tag.endswith('}v')),
            None,
        )
        if value_element is None:
            return _Cell(None, formula)
        raw = value_element.text or ''
        if cell_type == 's':
            try:
                return _Cell(self.shared_strings[int(raw)], formula)
            except (IndexError, ValueError):
                return _Cell(raw, formula)
        if cell_type == 'b':
            return _Cell(raw == '1', formula)
        if cell_type in {'str', 'e'}:
            return _Cell(raw, formula)
        try:
            number = float(raw)
            return _Cell(int(number) if number.is_integer() else number, formula)
        except ValueError:
            return _Cell(raw, formula)

    def _load_shared_strings(self) -> list[str]:
        if 'xl/sharedStrings.xml' not in self.names:
            return []
        root = ET.fromstring(self.archive.read('xl/sharedStrings.xml'))
        return [_xml_text(element) for element in root if element.tag.endswith('}si')]

    def _load_persons(self) -> dict[str, str]:
        output: dict[str, str] = {}
        for path in sorted(name for name in self.names if name.startswith('xl/persons/')):
            root = ET.fromstring(self.archive.read(path))
            for element in root.iter():
                if element.tag.endswith('}person') and element.get('id'):
                    output[element.get('id', '')] = element.get('displayName', '')
        return output

    def _load_sheet_comments(self, sheet_path: str) -> list[_RawComment]:
        sheet = PurePosixPath(sheet_path)
        rels_path = str(sheet.parent / '_rels' / f'{sheet.name}.rels')
        if rels_path not in self.names:
            return []
        output: list[_RawComment] = []
        for relationship in self._relationship_records(rels_path):
            target = self._normalize_target(str(sheet.parent), relationship['target'])
            relationship_type = relationship['type'].casefold()
            if 'threadedcomment' in relationship_type:
                output.extend(self._load_threaded_comments(target))
            elif relationship_type.endswith('/comments'):
                output.extend(self._load_legacy_comments(target))
        return output

    def _load_threaded_comments(self, path: str) -> list[_RawComment]:
        root = ET.fromstring(self.archive.read(path))
        output: list[_RawComment] = []
        for element in root.iter():
            if not element.tag.endswith('}threadedComment'):
                continue
            text = _xml_text(element)
            comment_id = element.get('id') or _comment_hash(
                element.get('ref', ''),
                text,
            )
            output.append(
                _RawComment(
                    comment_id=comment_id.strip('{}'),
                    cell_ref=element.get('ref', ''),
                    author=self.persons.get(element.get('personId', '')) or None,
                    text=text,
                    timestamp=_parse_datetime(element.get('dT')),
                    parent_comment_id=(element.get('parentId') or '').strip('{}') or None,
                    resolved=_parse_optional_bool(
                        element.get('done') or element.get('resolved')
                    ),
                    kind='threaded',
                )
            )
        return output

    def _load_legacy_comments(self, path: str) -> list[_RawComment]:
        root = ET.fromstring(self.archive.read(path))
        authors = [
            _xml_text(element)
            for element in root.iter()
            if element.tag.endswith('}author')
        ]
        output: list[_RawComment] = []
        for element in root.iter():
            if not element.tag.endswith('}comment'):
                continue
            author_id = int(element.get('authorId', '0'))
            author = authors[author_id] if author_id < len(authors) else ''
            if author.startswith('tc={'):
                continue
            text = _xml_text(element)
            ref = element.get('ref', '')
            output.append(
                _RawComment(
                    comment_id=_comment_hash(ref, author, text),
                    cell_ref=ref,
                    author=normalize_whitespace(author) or None,
                    text=text,
                    timestamp=None,
                    parent_comment_id=None,
                    resolved=None,
                    kind='note',
                )
            )
        return output

    def _relationships(self, path: str) -> dict[str, str]:
        return {
            record['id']: record['target']
            for record in self._relationship_records(path)
        }

    def _relationship_records(self, path: str) -> list[dict[str, str]]:
        root = ET.fromstring(self.archive.read(path))
        return [
            {
                'id': element.get('Id', ''),
                'target': element.get('Target', ''),
                'type': element.get('Type', ''),
            }
            for element in root.iter()
            if element.tag == f'{{{_PACKAGE_REL_NS}}}Relationship'
        ]

    @staticmethod
    def _normalize_target(base: str, target: str) -> str:
        if target.startswith('/'):
            return target.lstrip('/')
        return posixpath.normpath(posixpath.join(base, target))


def _xml_text(element: ET.Element | None) -> str:
    if element is None:
        return ''
    return normalize_whitespace(''.join(element.itertext()))


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.casefold() in {'1', 'true', 'yes'}


def _comment_hash(*values: str) -> str:
    payload = '\x1f'.join(values).encode('utf-8')
    return hashlib.sha1(payload).hexdigest()
