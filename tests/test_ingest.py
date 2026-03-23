import shutil
from pathlib import Path

from scibowl.ingest.question_sets import normalize_mit_question_csv
from scibowl.ingest.reviews import import_ratings_csv
from scibowl.utils.ids import make_id


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
