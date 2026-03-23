from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from pypdf import PdfReader

from scibowl.ingest.question_sets import _is_visual_bonus, _normalize_answer_mode, _normalize_category, _normalize_question_type
from scibowl.schema.common import SourceType
from scibowl.schema.common import AnswerMode
from scibowl.schema.question import Choice, NormalizedQuestion, Provenance, SourceMetadata
from scibowl.utils.ids import slugify
from scibowl.utils.text import normalize_whitespace


ENTRY_PATTERN = re.compile(
    r"""
    (?is)
    (?P<type>visual\s+bonus|bonus|toss(?:[\s-]*up)?)
    \s*
    (?P<number>\d+)[\.\)]
    \s*
    (?P<category>
        biology|
        chemistry|
        physics|
        math|
        energy|
        earth\s*(?:and|&)?\s*space|
        earth\s*science|
        earth|
        space|
        astronomy|
        machine\s*learning|
        statistics|
        theoretical\s*cs
    )
    \s*(?:-|:)?\s*
    (?P<format>multiple\s*choice|short\s*answer)
    \s*(?:-|:)?\s*
    (?P<question>.*?)
    \banswer\s*:\s*
    (?P<answer>.*?)
    (?=
        (?:
            visual\s+bonus|
            bonus|
            toss(?:[\s-]*up)?
        )\s*\d+[\.\)]
        |
        \Z
    )
    """,
    re.VERBOSE,
)

DECORATIVE_RUN_PATTERN = re.compile(r"(?:\s*[~*_=-]{3,}\s*)+")
WRITER_INITIALS_PATTERN = re.compile(r"\s*\[[A-Z]{1,4}\]\s*$")
PAGE_FOOTER_PATTERN = re.compile(
    r"""
    \s+
    [A-Za-z][A-Za-z0-9.'&/-]*
    \s+Science\s+Bowl
    (?:\s+Invitational)?
    (?:\s+Round\s+\d+)?
    (?:\s+(?:RR|DE|Finals?)\s+\d+)?
    \s+Page\s+\d+
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
CHOICE_LABEL_PATTERN = re.compile(r"(?<!\S)([WXYZ])\s*[\)\.]\s*", re.IGNORECASE)
MC_ANSWER_LABEL_PATTERN = re.compile(r"([WXYZ])(?:\s*[\)\.])?", re.IGNORECASE)


def extract_pdf_text(input_path: Path) -> str:
    pdftotext_bin = shutil.which("pdftotext")
    if pdftotext_bin:
        try:
            proc = subprocess.run(
                [pdftotext_bin, "-layout", str(input_path), "-"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )
            if proc.stdout.strip():
                return proc.stdout
        except (OSError, subprocess.SubprocessError):
            pass

    reader = PdfReader(str(input_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def parse_packet_pdf(
    input_path: Path,
    *,
    source_id: str,
    question_prefix: str | None = None,
    tournament: str | None = None,
) -> list[NormalizedQuestion]:
    return parse_packet_text(
        extract_pdf_text(input_path),
        source_id=source_id,
        question_prefix=question_prefix,
        tournament=tournament,
        raw_file=input_path,
    )


def parse_packet_text(
    text: str,
    *,
    source_id: str,
    question_prefix: str | None = None,
    tournament: str | None = None,
    raw_file: Path | None = None,
) -> list[NormalizedQuestion]:
    normalized = _normalize_packet_text(text)
    questions: list[NormalizedQuestion] = []
    question_index = 1
    id_prefix = question_prefix or source_id
    tournament_name = tournament or source_id

    for match in ENTRY_PATTERN.finditer(normalized):
        q_type = normalize_whitespace(match.group("type"))
        if _is_visual_bonus(q_type):
            continue

        answer_text = _clean_packet_answer(match.group("answer"))
        category_text = normalize_whitespace(match.group("category"))
        format_text = normalize_whitespace(match.group("format"))
        question_text = normalize_whitespace(match.group("question"))
        answer_mode = _normalize_answer_mode(format_text)
        choices: list[Choice] = []

        if answer_mode == AnswerMode.MULTIPLE_CHOICE:
            question_text, choices = _extract_packet_choices(question_text)
            if choices:
                answer_text = _normalize_packet_mc_answer(answer_text, choices)
            elif not MC_ANSWER_LABEL_PATTERN.fullmatch(answer_text):
                answer_mode = AnswerMode.SHORT_ANSWER

        try:
            category = _normalize_category(category_text)
        except ValueError:
            continue

        questions.append(
            NormalizedQuestion(
                question_id=f"{id_prefix}_{question_index:04d}",
                source_type=SourceType.PACKET,
                source_id=source_id,
                category=category,
                subcategory="other",
                question_type=_normalize_question_type(q_type),
                answer_mode=answer_mode,
                difficulty=3,
                question_text=question_text,
                answer_text=f"ANSWER: {answer_text}",
                choices=choices,
                source_metadata=SourceMetadata(
                    source_row=question_index,
                    tournament=tournament_name,
                    original_difficulty=None,
                    original_quality=None,
                ),
                provenance=Provenance(
                    raw_file=str(raw_file) if raw_file else None,
                    parser_version="packet_pdf_v3",
                    review_status="unreviewed",
                ),
            )
        )
        question_index += 1

    return questions


def _normalize_packet_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    normalized = re.sub(r"(?i)\bmultiplechoice\b", "Multiple Choice", normalized)
    normalized = re.sub(r"(?i)\bshortanswer\b", "Short Answer", normalized)
    normalized = re.sub(r"(?i)\btossup\b", "Tossup", normalized)
    normalized = re.sub(r"(?i)\btoss-up\b", "Toss-Up", normalized)
    normalized = re.sub(r"(?i)\bvisualbonus\b", "Visual Bonus", normalized)
    normalized = re.sub(r"(?i)\bbonus(\d+[.)])", r"Bonus \1", normalized)
    normalized = re.sub(r"(?i)\btoss(?:[\s-]*up)?(\d+[.)])", r"Tossup \1", normalized)
    return normalized


def _clean_packet_answer(text: str) -> str:
    cleaned = normalize_whitespace(text)
    cleaned = DECORATIVE_RUN_PATTERN.sub(" ", cleaned)
    cleaned = PAGE_FOOTER_PATTERN.sub("", cleaned)
    cleaned = WRITER_INITIALS_PATTERN.sub("", cleaned)
    return normalize_whitespace(cleaned)


def _extract_packet_choices(question_text: str) -> tuple[str, list[Choice]]:
    matches = list(CHOICE_LABEL_PATTERN.finditer(question_text))
    if len(matches) < 4:
        return question_text, []

    sequence = _find_choice_sequence(matches)
    if sequence is None:
        return question_text, []

    stem = normalize_whitespace(question_text[: sequence[0].start()])
    choices: list[Choice] = []

    for index, match in enumerate(sequence):
        start = match.end()
        end = sequence[index + 1].start() if index < 3 else len(question_text)
        choice_text = normalize_whitespace(question_text[start:end])
        if not choice_text:
            return question_text, []
        choices.append(Choice(label=match.group(1).upper(), text=choice_text))

    return stem, choices


def _find_choice_sequence(matches: list[re.Match[str]]) -> list[re.Match[str]] | None:
    for start_index, match in enumerate(matches):
        if match.group(1).upper() != "W":
            continue

        sequence = [match]
        next_from = start_index + 1
        for label in ["X", "Y", "Z"]:
            next_match = next(
                (candidate for candidate in matches[next_from:] if candidate.group(1).upper() == label),
                None,
            )
            if next_match is None:
                sequence = []
                break
            sequence.append(next_match)
            next_from = matches.index(next_match) + 1

        if len(sequence) == 4:
            return sequence

    return None


def _normalize_packet_mc_answer(answer_text: str, choices: list[Choice]) -> str:
    normalized = normalize_whitespace(answer_text)
    match = re.fullmatch(r"([WXYZ])(?:\s*[\)\.\]:-])?", normalized, flags=re.IGNORECASE)
    if not match:
        return normalized

    label = match.group(1).upper()
    choice = next((item for item in choices if item.label == label), None)
    if choice is None:
        return normalized
    return f"{choice.label}) {choice.text}"


def parse_packet_directory(
    input_dir: Path,
    *,
    source_id: str | None = None,
) -> tuple[list[NormalizedQuestion], list[dict[str, object]]]:
    tournament_id = source_id or slugify(input_dir.name)
    all_questions: list[NormalizedQuestion] = []
    report: list[dict[str, object]] = []

    for pdf_path in sorted(input_dir.glob("*.pdf")):
        packet_prefix = f"{tournament_id}__{slugify(pdf_path.stem)}"
        questions = parse_packet_pdf(
            pdf_path,
            source_id=tournament_id,
            question_prefix=packet_prefix,
            tournament=tournament_id,
        )
        all_questions.extend(questions)
        report.append(
            {
                "file": str(pdf_path),
                "question_count": len(questions),
            }
        )

    return all_questions, report
