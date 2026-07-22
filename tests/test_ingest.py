import shutil
from pathlib import Path

from scibowl.ingest.textbooks import _clean_textbook_page, _trim_pages_after_contents, _trim_pages_before_back_matter
from scibowl.ingest.question_sets import normalize_mit_question_csv
from scibowl.ingest.reviews import import_ratings_csv
from scibowl.utils.ids import make_id
from scibowl.utils.text import chunk_paragraphs, normalize_paragraphs


def _make_temp_dir() -> Path:
    path = Path("tests_runtime") / make_id("case")
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_normalize_mit_csv_preserves_guidance() -> None:
    tmp_path = _make_temp_dir()
    csv_path = tmp_path / "mit.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Type,Category,Format,Question,W,X,Y,Z,Answer,Accept,Do Not Accept,Subcategory,Writer,Source,Date,Division (Approx),Round",
                '"Toss-up","Biology","Short Answer","What mutation shifts a reading frame?","","","","","Frameshift Mutation","frameshift","substitution","Cell Biology","Sean","Campbell","5/10/2025","RR","1"',
            ]
        ),
        encoding="utf-8",
    )

    rows = normalize_mit_question_csv(csv_path, source_id="mit_2025")

    assert len(rows) == 1
    assert rows[0].answer_guidance.accept == ["frameshift"]
    assert rows[0].answer_guidance.do_not_accept == ["substitution"]
    assert rows[0].source_metadata.writer == "Sean"
    assert rows[0].source_metadata.division == "RR"
    assert rows[0].source_metadata.source_row == 1
    shutil.rmtree(tmp_path)


def test_import_ratings_csv_links_by_source_row() -> None:
    tmp_path = _make_temp_dir()
    questions_csv = tmp_path / "mit.csv"
    questions_csv.write_text(
        "\n".join(
            [
                "Type,Category,Format,Question,W,X,Y,Z,Answer,Accept,Do Not Accept,Subcategory,Writer,Source,Date,Division (Approx),Round",
                '"Toss-up","Biology","Short Answer","What mutation shifts a reading frame?","","","","","Frameshift Mutation","","","Cell Biology","Sean","Campbell","5/10/2025","RR","1"',
            ]
        ),
        encoding="utf-8",
    )
    questions = normalize_mit_question_csv(questions_csv, source_id="mit_2025")

    questions_jsonl = tmp_path / "questions.jsonl"
    questions_jsonl.write_text("\n".join(item.model_dump_json() for item in questions) + "\n", encoding="utf-8")

    ratings_csv = tmp_path / "ratings.csv"
    ratings_csv.write_text(
        "\n".join(
            [
                "Source Row,Reviewer,Quality Rating,Difficulty Rating",
                '"1","Alice",4,3',
            ]
        ),
        encoding="utf-8",
    )

    reviews = import_ratings_csv(ratings_csv, questions_path=questions_jsonl, tournament="MIT Science Bowl 2025")

    assert len(reviews) == 1
    assert reviews[0].question_id == "mit_2025_0001"
    assert reviews[0].ratings.quality == 4
    assert reviews[0].ratings.difficulty == 3
    assert reviews[0].reviewer_id == "alice"
    shutil.rmtree(tmp_path)


def test_normalize_mit_csv_skips_visual_bonus() -> None:
    tmp_path = _make_temp_dir()
    csv_path = tmp_path / "mit.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Type,Category,Format,Question,W,X,Y,Z,Answer,Accept,Do Not Accept,Subcategory,Writer,Source,Date,Division (Approx),Round",
                '"Visual Bonus","Biology","Short Answer","Identify the organelle in the image","","","","","Mitochondrion","","","Cell Biology","Sean","Campbell","5/10/2025","RR","1"',
                '"Toss-up","Biology","Short Answer","What mutation shifts a reading frame?","","","","","Frameshift Mutation","","","Cell Biology","Sean","Campbell","5/10/2025","RR","1"',
            ]
        ),
        encoding="utf-8",
    )

    rows = normalize_mit_question_csv(csv_path, source_id="mit_2025")

    assert len(rows) == 1
    assert "visual" not in rows[0].question_text.lower()
    shutil.rmtree(tmp_path)


def test_trim_pages_after_contents_skips_front_matter() -> None:
    pages = [
        "Preface\nThis book comes with a companion website.",
        "Contents\nChapter 1 .... 1\nChapter 2 .... 35\nAppendix .... 700",
        "Chapter 1\nMatter consists of atoms and molecules.",
        "Chapter 1 continued\nEnergy is conserved in closed systems.",
    ]

    trimmed_pages, metadata = _trim_pages_after_contents(pages)

    assert trimmed_pages[0].startswith("Chapter 1")
    assert metadata["front_matter_trimmed"] is True
    assert metadata["content_start_page"] == 3


def test_trim_pages_before_back_matter_skips_glossary_and_index() -> None:
    pages = [
        "Chapter 10\nGroundwater moves through permeable material.",
        "Chapter 11\nCold fronts form when advancing cold air undercuts warm air.",
        "Glossary\nAquifer: a body of rock or sediment that stores groundwater.",
        "Index\natmosphere, 10\naquifer, 55",
    ]

    trimmed_pages, metadata = _trim_pages_before_back_matter(pages)

    assert len(trimmed_pages) == 2
    assert trimmed_pages[-1].startswith("Chapter 11")
    assert metadata["back_matter_trimmed"] is True
    assert metadata["back_matter_start_page"] == 3


def test_clean_textbook_page_strips_exercises_and_key_terms() -> None:
    page = "\n".join(
        [
            "CHAPTER 17 | GROUNDWATER",
            "Groundwater flows through permeable sediments.",
            "Q Examine this cross section and label the aquifer.",
            "Key Terms porosity permeability aquifer aquitard",
            "Problem 3.1 The luminosity of Vega is ...",
            "",
            "Streams may gain water from groundwater inflow.",
        ]
    )

    cleaned = _clean_textbook_page(page)

    assert "label the aquifer" not in cleaned
    assert "Key Terms" not in cleaned
    assert "Problem 3.1" not in cleaned
    assert "Groundwater flows through permeable sediments." in cleaned
    assert "Streams may gain water from groundwater inflow." in cleaned


def test_clean_textbook_page_drops_interleaved_review_page() -> None:
    text = (
        'SUMMARY OF KEY CONCEPTS 5 Chapter Review CONCEPT 5.1 Macromolecules are polymers. '
        'Levels 1-2: Remembering/Understanding 1. Which molecule is a polymer?'
    )

    assert _clean_textbook_page(text) == ''


def test_clean_textbook_page_keeps_text_after_inline_chapter_contents() -> None:
    text = (
        'Overview Vertical Structure Weather and Climate Summary Questions for Review '
        'Questions for Thought and Exploration Contents '
        'The atmosphere contains gases that absorb and emit radiation. '
        'Pressure and density both decrease with altitude through most of the atmosphere. '
        'Weather describes short-term atmospheric conditions, while climate describes long-term patterns. '
        'These ideas support the scientific study of storms and circulation.'
    )

    cleaned = _clean_textbook_page(text)

    assert cleaned.startswith('The atmosphere contains gases')
    assert 'Questions for Review' not in cleaned


def test_chunk_paragraphs_preserves_paragraph_boundaries() -> None:
    text = "\n\n".join(
        [
            "Paragraph one explains aquifers and permeability in groundwater flow.",
            "Paragraph two explains gaining streams and losing streams in a watershed.",
            "Paragraph three explains porosity and specific yield.",
        ]
    )

    paragraphs = normalize_paragraphs(text)
    chunks = chunk_paragraphs(paragraphs, max_words=14, overlap_words=4)

    assert len(chunks) >= 2
    assert "Paragraph one explains aquifers" in chunks[0]
    assert "Paragraph two explains gaining streams" in chunks[1] or "Paragraph three explains" in chunks[1]
