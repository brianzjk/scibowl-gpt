from __future__ import annotations

import html
import shutil
import zipfile
from pathlib import Path

from scibowl.ingest.mit_workbooks import normalize_mit_writing_workbooks
from scibowl.utils.ids import make_id


def _make_temp_dir() -> Path:
    path = Path('tests_runtime') / make_id('mit_xlsx')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _inline_cell(ref: str, value: str) -> str:
    return (
        f'<c r="{ref}" t="inlineStr"><is><t>{html.escape(value)}</t></is></c>'
    )


def _number_cell(ref: str, value: float) -> str:
    return f'<c r="{ref}"><v>{value}</v></c>'


def _formula_cell(ref: str, formula: str, cached: float) -> str:
    return f'<c r="{ref}"><f>{formula}</f><v>{cached}</v></c>'


def _write_fixture(path: Path) -> None:
    headers = [
        'Type',
        'Category',
        'Format',
        'Question',
        'W',
        'X',
        'Y',
        'Z',
        'Answer',
        'Accept',
        'Do Not Accept',
        'Subcategory',
        'Writer',
        'Difficulty',
        'Quality',
        'D1',
        'D2',
        'Q1',
        'Q2',
    ]
    row1 = ''.join(
        _inline_cell(f'{column}1', value)
        for column, value in zip(
            [
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
            ],
            headers,
            strict=True,
        )
    )
    row2 = ''.join(
        [
            _inline_cell('A2', 'Bonus'),
            _inline_cell('B2', 'Biology'),
            _inline_cell('C2', 'Multiple Choice'),
            _inline_cell('D2', 'Which organelle makes most cellular ATP?'),
            _inline_cell('E2', 'Nucleus'),
            _inline_cell('F2', 'Mitochondrion'),
            _inline_cell('G2', 'Lysosome'),
            _inline_cell('H2', 'Golgi apparatus'),
            _inline_cell('I2', 'X'),
            _inline_cell('L2', 'Cell Biology'),
            _inline_cell('M2', 'Writer A'),
            _formula_cell('N2', 'AVERAGE(P2:Q2)', 4.5),
            _formula_cell('O2', 'AVERAGE(R2:S2)', 0.5),
            _number_cell('P2', 4),
            _number_cell('Q2', 5),
            _number_cell('R2', 1),
            _number_cell('S2', 0),
        ]
    )
    row3 = ''.join(
        [
            _inline_cell('A3', 'Toss-up'),
            _inline_cell('B3', 'Biology'),
            _inline_cell('C3', 'Short Answer'),
            _inline_cell('D3', 'What molecule carries genetic information?'),
            _inline_cell('I3', 'DNA'),
            _formula_cell('N3', 'AVERAGE(P3:Q3)', 100),
            _formula_cell('O3', 'AVERAGE(R3:S3)', 5),
            _number_cell('P3', 100),
        ]
    )
    row4 = ''.join(
        [
            _inline_cell('A4', 'Visual Bonus'),
            _inline_cell('B4', 'Biology'),
            _inline_cell('C4', 'Short Answer'),
            _inline_cell('D4', 'Identify the pictured organelle.'),
            _inline_cell('I4', 'Mitochondrion'),
        ]
    )
    worksheet = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData><row r="1">{row1}</row><row r="2">{row2}</row>'
        f'<row r="3">{row3}</row><row r="4">{row4}</row></sheetData>'
        '</worksheet>'
    )
    workbook = (
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Template" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    workbook_rels = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    sheet_rels = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" '
        'Target="../comments1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.microsoft.com/office/2017/10/relationships/threadedComment" '
        'Target="../threadedComments/threadedComment1.xml"/></Relationships>'
    )
    threaded = (
        '<ThreadedComments xmlns="http://schemas.microsoft.com/office/spreadsheetml/2018/threadedcomments">'
        '<threadedComment ref="P1" personId="p1" id="rating-d"><text>Alice</text></threadedComment>'
        '<threadedComment ref="R1" personId="p1" id="rating-q"><text>Alice</text></threadedComment>'
        '<threadedComment ref="D2" personId="p1" id="comment-1"><text>Good clue</text></threadedComment>'
        '<threadedComment ref="D2" personId="p2" id="comment-2" parentId="comment-1"><text>Agreed</text></threadedComment>'
        '</ThreadedComments>'
    )
    persons = (
        '<personList xmlns="http://schemas.microsoft.com/office/spreadsheetml/2018/threadedcomments">'
        '<person id="p1" displayName="Alice Reviewer"/>'
        '<person id="p2" displayName="Bob Reviewer"/>'
        '</personList>'
    )
    legacy = (
        '<comments xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<authors><author>Editor</author><author>tc={comment-1}</author></authors>'
        '<commentList>'
        '<comment ref="D3" authorId="0"><text><t>Needs a source</t></text></comment>'
        '<comment ref="D2" authorId="1"><text><t>Compatibility copy</t></text></comment>'
        '</commentList></comments>'
    )
    core = (
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dcterms="http://purl.org/dc/terms/">'
        '<dcterms:modified>2026-07-21T12:00:00Z</dcterms:modified></cp:coreProperties>'
    )
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('xl/workbook.xml', workbook)
        archive.writestr('xl/_rels/workbook.xml.rels', workbook_rels)
        archive.writestr('xl/worksheets/sheet1.xml', worksheet)
        archive.writestr('xl/worksheets/_rels/sheet1.xml.rels', sheet_rels)
        archive.writestr('xl/threadedComments/threadedComment1.xml', threaded)
        archive.writestr('xl/persons/person.xml', persons)
        archive.writestr('xl/comments1.xml', legacy)
        archive.writestr('docProps/core.xml', core)


def test_normalize_mit_workbooks_preserves_ratings_and_comments() -> None:
    tmp_path = _make_temp_dir()
    workbook_path = tmp_path / '2026 Biology Question Writing.xlsx'
    _write_fixture(workbook_path)

    imported = normalize_mit_writing_workbooks(tmp_path)
    by_row = {question.source_metadata.source_row: question for question in imported.questions}

    assert set(by_row) == {2, 3}
    assert by_row[2].source_metadata.original_difficulty == 4.5
    assert by_row[2].source_metadata.original_quality == 0.5
    assert by_row[3].source_metadata.original_difficulty is None
    assert by_row[3].source_metadata.original_quality is None
    assert by_row[2].answer_text == 'ANSWER: X) Mitochondrion'
    assert len(imported.reviews) == 4
    assert imported.reviews[0].reviewer_id == 'alice'

    comments_by_text = {comment.text: comment for comment in imported.comments}
    assert set(comments_by_text) == {
        'Alice',
        'Good clue',
        'Agreed',
        'Needs a source',
    }
    assert comments_by_text['Good clue'].question_id == by_row[2].question_id
    assert comments_by_text['Needs a source'].question_id == by_row[3].question_id
    assert imported.summary['files'][0]['invalid_rating_count'] == 1
    assert imported.summary['files'][0]['skip_counts']['visual_bonus'] == 1
    shutil.rmtree(tmp_path)
