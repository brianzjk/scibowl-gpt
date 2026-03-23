from __future__ import annotations

import argparse
from pathlib import Path

from scibowl.eval.baseline import run_baseline_eval
from scibowl.eval.benchmark import build_evaluation_record
from scibowl.eval.splits import build_rated_question_split
from scibowl.ingest.nsb_samples import download_sample_packets
from scibowl.generate.orchestration import GenerationOrchestrator
from scibowl.ingest.normalize import validate_normalized_questions
from scibowl.ingest.packets import parse_packet_directory, parse_packet_pdf
from scibowl.ingest.question_sets import normalize_mit_question_csv, normalize_mit_writing_directory
from scibowl.ingest.reviews import import_mit_rating_directory, import_ratings_csv
from scibowl.ingest.textbooks import ingest_textbook_text
from scibowl.schema.generation import QuestionSpec
from scibowl.schema.question import NormalizedQuestion
from scibowl.schema.textbook import TextbookChunk
from scibowl.utils.io import read_json, read_jsonl, write_json, write_jsonl


def cmd_ingest_textbook(args: argparse.Namespace) -> None:
    chunks = ingest_textbook_text(Path(args.input_path), document_id=args.document_id, title=args.title)
    write_jsonl(Path(args.output_path), chunks)
    print(f"Wrote {len(chunks)} textbook chunks to {args.output_path}")


def cmd_ingest_packet_pdf(args: argparse.Namespace) -> None:
    rows = parse_packet_pdf(Path(args.input_path), source_id=args.source_id)
    write_jsonl(Path(args.output_path), rows)
    print(f"Wrote {len(rows)} normalized packet questions to {args.output_path}")


def cmd_ingest_packet_dir(args: argparse.Namespace) -> None:
    rows, report = parse_packet_directory(Path(args.input_dir), source_id=args.source_id)
    write_jsonl(Path(args.output_path), rows)
    if args.report_path:
        write_json(Path(args.report_path), {"files": report, "total_questions": len(rows)})
    print(f"Wrote {len(rows)} normalized packet questions to {args.output_path}")


def cmd_download_nsb_hs_samples(args: argparse.Namespace) -> None:
    paths = download_sample_packets(Path(args.output_dir), min_set=args.min_set)
    print(f"Downloaded {len(paths)} packet PDFs to {args.output_dir}")


def cmd_validate_questions(args: argparse.Namespace) -> None:
    rows = validate_normalized_questions(Path(args.input_path))
    print(f"Validated {len(rows)} normalized questions")


def cmd_normalize_mit_csv(args: argparse.Namespace) -> None:
    rows = normalize_mit_question_csv(Path(args.input_path), source_id=args.source_id)
    write_jsonl(Path(args.output_path), rows)
    print(f"Wrote {len(rows)} normalized MIT questions to {args.output_path}")


def cmd_normalize_mit_writing_dir(args: argparse.Namespace) -> None:
    rows = normalize_mit_writing_directory(Path(args.input_dir), source_id=args.source_id)
    write_jsonl(Path(args.output_path), rows)
    print(f"Wrote {len(rows)} normalized MIT writing-sheet questions to {args.output_path}")


def cmd_import_ratings_csv(args: argparse.Namespace) -> None:
    rows = import_ratings_csv(
        Path(args.input_path),
        questions_path=Path(args.questions) if args.questions else None,
        tournament=args.tournament,
    )
    write_jsonl(Path(args.output_path), rows)
    print(f"Wrote {len(rows)} human reviews to {args.output_path}")


def cmd_import_mit_ratings_dir(args: argparse.Namespace) -> None:
    rows = import_mit_rating_directory(
        Path(args.input_dir),
        questions_path=Path(args.questions),
        tournament=args.tournament,
    )
    write_jsonl(Path(args.output_path), rows)
    print(f"Wrote {len(rows)} MIT review ratings to {args.output_path}")


def cmd_build_rated_split(args: argparse.Namespace) -> None:
    split = build_rated_question_split(
        Path(args.questions),
        Path(args.reviews),
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    write_json(Path(args.output_path), split)
    print(
        "Built rated split "
        f"(train={split['counts']['train']}, val={split['counts']['val']}, test={split['counts']['test']})"
    )


def cmd_run_baseline_eval(args: argparse.Namespace) -> None:
    records = run_baseline_eval(
        split_path=Path(args.split),
        split_name=args.split_name,
        output_dir=Path(args.output_dir),
        questions_path=Path(args.questions) if args.questions else None,
        reviews_path=Path(args.reviews) if args.reviews else None,
        textbook_chunks_path=Path(args.textbook_chunks) if args.textbook_chunks else None,
        max_items=args.max_items,
    )
    print(f"Wrote {len(records)} baseline evaluation records to {args.output_dir}")


def cmd_demo_generate(args: argparse.Namespace) -> None:
    spec = QuestionSpec.model_validate(read_json(Path(args.spec)))
    textbook_chunks = read_jsonl(Path(args.textbook_chunks), TextbookChunk)
    style_questions = read_jsonl(Path(args.style_questions), NormalizedQuestion)

    orchestrator = GenerationOrchestrator()
    bundle, draft, report = orchestrator.run(spec, textbook_chunks, style_questions)
    eval_record = build_evaluation_record("demo_run", draft.draft_id, spec, draft, report)

    output_dir = Path(args.output_dir)
    write_json(output_dir / "retrieval_bundle.json", bundle.model_dump())
    write_json(output_dir / "draft.json", draft.model_dump())
    write_json(output_dir / "report.json", report.model_dump())
    write_json(output_dir / "evaluation.json", eval_record.model_dump())
    print(f"Wrote generation artifacts to {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scibowl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_textbook = subparsers.add_parser("ingest-textbook")
    ingest_textbook.add_argument("input_path")
    ingest_textbook.add_argument("output_path")
    ingest_textbook.add_argument("--document-id")
    ingest_textbook.add_argument("--title")
    ingest_textbook.set_defaults(func=cmd_ingest_textbook)

    ingest_packet_pdf = subparsers.add_parser("ingest-packet-pdf")
    ingest_packet_pdf.add_argument("input_path")
    ingest_packet_pdf.add_argument("output_path")
    ingest_packet_pdf.add_argument("--source-id", required=True)
    ingest_packet_pdf.set_defaults(func=cmd_ingest_packet_pdf)

    ingest_packet_dir = subparsers.add_parser("ingest-packet-dir")
    ingest_packet_dir.add_argument("input_dir")
    ingest_packet_dir.add_argument("output_path")
    ingest_packet_dir.add_argument("--source-id")
    ingest_packet_dir.add_argument("--report-path")
    ingest_packet_dir.set_defaults(func=cmd_ingest_packet_dir)

    download_nsb_hs_samples = subparsers.add_parser("download-nsb-hs-samples")
    download_nsb_hs_samples.add_argument("--output-dir", required=True)
    download_nsb_hs_samples.add_argument("--min-set", type=int, default=13)
    download_nsb_hs_samples.set_defaults(func=cmd_download_nsb_hs_samples)

    validate_questions = subparsers.add_parser("validate-questions")
    validate_questions.add_argument("input_path")
    validate_questions.set_defaults(func=cmd_validate_questions)

    normalize_mit_csv = subparsers.add_parser("normalize-mit-csv")
    normalize_mit_csv.add_argument("input_path")
    normalize_mit_csv.add_argument("output_path")
    normalize_mit_csv.add_argument("--source-id", default="mit_2025")
    normalize_mit_csv.set_defaults(func=cmd_normalize_mit_csv)

    normalize_mit_writing_dir = subparsers.add_parser("normalize-mit-writing-dir")
    normalize_mit_writing_dir.add_argument("input_dir")
    normalize_mit_writing_dir.add_argument("output_path")
    normalize_mit_writing_dir.add_argument("--source-id", default="mit_2025")
    normalize_mit_writing_dir.set_defaults(func=cmd_normalize_mit_writing_dir)

    import_ratings_csv = subparsers.add_parser("import-ratings-csv")
    import_ratings_csv.add_argument("input_path")
    import_ratings_csv.add_argument("output_path")
    import_ratings_csv.add_argument("--questions")
    import_ratings_csv.add_argument("--tournament")
    import_ratings_csv.set_defaults(func=cmd_import_ratings_csv)

    import_mit_ratings_dir = subparsers.add_parser("import-mit-ratings-dir")
    import_mit_ratings_dir.add_argument("input_dir")
    import_mit_ratings_dir.add_argument("output_path")
    import_mit_ratings_dir.add_argument("--questions", required=True)
    import_mit_ratings_dir.add_argument("--tournament", default="MIT Science Bowl 2025")
    import_mit_ratings_dir.set_defaults(func=cmd_import_mit_ratings_dir)

    build_rated_split = subparsers.add_parser("build-rated-split")
    build_rated_split.add_argument("--questions", required=True)
    build_rated_split.add_argument("--reviews", required=True)
    build_rated_split.add_argument("--output-path", required=True)
    build_rated_split.add_argument("--seed", type=int, default=42)
    build_rated_split.add_argument("--train-fraction", type=float, default=0.8)
    build_rated_split.add_argument("--val-fraction", type=float, default=0.1)
    build_rated_split.set_defaults(func=cmd_build_rated_split)

    run_baseline_eval_cmd = subparsers.add_parser("run-baseline-eval")
    run_baseline_eval_cmd.add_argument("--split", required=True)
    run_baseline_eval_cmd.add_argument("--split-name", default="test")
    run_baseline_eval_cmd.add_argument("--output-dir", required=True)
    run_baseline_eval_cmd.add_argument("--questions")
    run_baseline_eval_cmd.add_argument("--reviews")
    run_baseline_eval_cmd.add_argument("--textbook-chunks")
    run_baseline_eval_cmd.add_argument("--max-items", type=int)
    run_baseline_eval_cmd.set_defaults(func=cmd_run_baseline_eval)

    demo_generate = subparsers.add_parser("demo-generate")
    demo_generate.add_argument("--spec", required=True)
    demo_generate.add_argument("--textbook-chunks", required=True)
    demo_generate.add_argument("--style-questions", required=True)
    demo_generate.add_argument("--output-dir", required=True)
    demo_generate.set_defaults(func=cmd_demo_generate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
