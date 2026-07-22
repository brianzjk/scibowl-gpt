from __future__ import annotations

import argparse
from pathlib import Path

from scibowl.eval.baseline import run_baseline_eval
from scibowl.eval.benchmark import build_evaluation_record
from scibowl.eval.splits import build_rated_question_split
from scibowl.generate.batch import run_generation_batch
from scibowl.generate.orchestration import GenerationOrchestrator
from scibowl.generation_review.server import run_generated_question_review_server
from scibowl.ingest.normalize import validate_normalized_questions
from scibowl.ingest.mit_workbooks import normalize_mit_writing_workbooks
from scibowl.ingest.question_sets import normalize_mit_question_csv, normalize_mit_writing_directory
from scibowl.ingest.reviews import import_mit_rating_directory, import_ratings_csv
from scibowl.ingest.textbook_corpus import ingest_textbook_corpus, load_textbook_chunks
from scibowl.ingest.textbooks import ingest_textbook_text
from scibowl.llm.runtime import apply_runtime_model_overrides
from scibowl.schema.generation import QuestionSpec
from scibowl.schema.question import NormalizedQuestion
from scibowl.schema.textbook import TextbookChunk
from scibowl.training import CLEAN_SFT_PROFILES, build_clean_sft_dataset, export_sft_dataset
from scibowl.training.lora import train_lora_from_config
from scibowl.utils.io import read_json, read_jsonl, write_json, write_jsonl


def cmd_ingest_textbook(args: argparse.Namespace) -> None:
    chunks = ingest_textbook_text(
        Path(args.input_path),
        document_id=args.document_id,
        title=args.title,
        topics=args.topic,
        start_page=args.start_page,
        end_page=args.end_page,
        max_words=args.max_words,
        overlap_words=args.overlap_words,
        corpus_version=args.corpus_version,
        expected_sha256=args.expected_sha256,
        expected_page_count=args.expected_page_count,
    )
    write_jsonl(Path(args.output_path), chunks)
    print(f"Wrote {len(chunks)} textbook chunks to {args.output_path}")


def cmd_ingest_textbook_corpus(args: argparse.Namespace) -> None:
    summary = ingest_textbook_corpus(
        Path(args.manifest),
        Path(args.raw_dir),
        Path(args.output_dir),
    )
    total_chunks = summary['total_chunks']
    source_count = summary['source_count']
    print(
        f'Wrote {total_chunks} chunks from {source_count} textbooks to {args.output_dir}'
    )


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


def cmd_normalize_mit_writing_xlsx_dir(args: argparse.Namespace) -> None:
    output_path = Path(args.output_path)
    reviews_output = (
        Path(args.reviews_output)
        if args.reviews_output
        else output_path.with_name(f'{output_path.stem}_reviews.jsonl')
    )
    comments_output = (
        Path(args.comments_output)
        if args.comments_output
        else output_path.with_name(f'{output_path.stem}_comments.jsonl')
    )
    manifest_output = (
        Path(args.manifest_output)
        if args.manifest_output
        else output_path.with_name(f'{output_path.stem}_manifest.json')
    )
    imported = normalize_mit_writing_workbooks(
        Path(args.input_dir),
        source_id=args.source_id,
        tournament=args.tournament,
        year=args.year,
    )
    write_jsonl(output_path, imported.questions)
    write_jsonl(reviews_output, imported.reviews)
    write_jsonl(comments_output, imported.comments)
    write_json(manifest_output, imported.summary)
    print(
        f'Wrote {len(imported.questions)} questions, {len(imported.reviews)} ratings, '
        f'and {len(imported.comments)} source comments'
    )


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
    _apply_model_override_args(args)
    records = run_baseline_eval(
        split_path=Path(args.split),
        split_name=args.split_name,
        output_dir=Path(args.output_dir),
        questions_path=Path(args.questions) if args.questions else None,
        style_questions_path=Path(args.style_questions) if args.style_questions else None,
        reviews_path=Path(args.reviews) if args.reviews else None,
        textbook_chunks_path=Path(args.textbook_chunks) if args.textbook_chunks else None,
        max_items=args.max_items,
        max_per_category=args.max_per_category,
        excluded_categories=args.exclude_category,
    )
    print(f"Wrote {len(records)} baseline evaluation records to {args.output_dir}")


def cmd_demo_generate(args: argparse.Namespace) -> None:
    _apply_model_override_args(args)
    spec = QuestionSpec.model_validate(read_json(Path(args.spec)))
    textbook_chunks = _load_textbook_chunks_arg(Path(args.textbook_chunks))
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


def _load_textbook_chunks_arg(path: Path) -> list[TextbookChunk]:
    return load_textbook_chunks(path)


def cmd_review_generated_questions(args: argparse.Namespace) -> None:
    output_path = Path(args.output_path) if args.output_path else Path(args.runs_path).with_name(
        Path(args.runs_path).stem + "_reviews.jsonl"
    )
    run_generated_question_review_server(
        runs_path=Path(args.runs_path),
        output_path=output_path,
        reviewer_id=args.reviewer_id,
        host=args.host,
        port=args.port,
        title=args.title,
    )


def cmd_generate_from_config(args: argparse.Namespace) -> None:
    _apply_model_override_args(args)
    records = run_generation_batch(Path(args.config_path))
    print(f"Wrote {len(records)} generated question runs from {args.config_path}")


def cmd_export_sft_dataset(args: argparse.Namespace) -> None:
    summary = export_sft_dataset(
        questions_path=Path(args.questions),
        output_dir=Path(args.output_dir),
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        excluded_categories=tuple(args.exclude_category),
    )
    print(
        "Exported SFT dataset "
        f"(unique={summary['unique_questions']}, train={summary['split_counts']['train']}, "
        f"val={summary['split_counts']['val']}, materialized_train={summary['materialized_train_count']})"
    )


def cmd_train_lora_sft(args: argparse.Namespace) -> None:
    train_lora_from_config(Path(args.config_path))


def cmd_build_clean_sft_dataset(args: argparse.Namespace) -> None:
    summary = build_clean_sft_dataset(
        questions_path=tuple(Path(value) for value in args.questions),
        output_dir=Path(args.output_dir),
        review_paths=tuple(Path(value) for value in args.review),
        profile=args.profile,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        held_out_question_ids_path=(
            Path(args.held_out_question_ids) if args.held_out_question_ids else None
        ),
        minimum_writing_quality=args.minimum_writing_quality,
        include_unrated_writing=args.include_unrated_writing,
    )
    counts = summary['split_counts']
    train_count = counts['train']
    val_count = counts['val']
    test_count = counts['test']
    held_out_count = counts['held_out']
    print(
        f'Built clean SFT dataset (train={train_count}, val={val_count}, '
        f'test={test_count}, held_out={held_out_count})'
    )


def _add_model_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ollama-model")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--writer-model")
    parser.add_argument("--writer-base-url")
    parser.add_argument("--verifier-model")
    parser.add_argument("--verifier-base-url")
    parser.add_argument("--writer-timeout-seconds", type=int)
    parser.add_argument("--verifier-timeout-seconds", type=int)
    parser.add_argument("--disable-writer-fallback", action="store_true")


def _apply_model_override_args(args: argparse.Namespace) -> None:
    apply_runtime_model_overrides(
        ollama_model=getattr(args, "ollama_model", None),
        ollama_base_url=getattr(args, "ollama_base_url", None),
        writer_model=getattr(args, "writer_model", None),
        writer_base_url=getattr(args, "writer_base_url", None),
        verifier_model=getattr(args, "verifier_model", None),
        verifier_base_url=getattr(args, "verifier_base_url", None),
        writer_timeout_seconds=getattr(args, "writer_timeout_seconds", None),
        verifier_timeout_seconds=getattr(args, "verifier_timeout_seconds", None),
        disable_writer_fallback=getattr(args, "disable_writer_fallback", False),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scibowl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_textbook = subparsers.add_parser("ingest-textbook")
    ingest_textbook.add_argument("input_path")
    ingest_textbook.add_argument("output_path")
    ingest_textbook.add_argument("--document-id")
    ingest_textbook.add_argument("--title")
    ingest_textbook.add_argument('--topic', action='append', default=[])
    ingest_textbook.add_argument('--start-page', type=int)
    ingest_textbook.add_argument('--end-page', type=int)
    ingest_textbook.add_argument('--max-words', type=int, default=180)
    ingest_textbook.add_argument('--overlap-words', type=int, default=30)
    ingest_textbook.add_argument('--corpus-version', default='textbook_v2')
    ingest_textbook.add_argument('--expected-sha256')
    ingest_textbook.add_argument('--expected-page-count', type=int)
    ingest_textbook.set_defaults(func=cmd_ingest_textbook)

    ingest_corpus = subparsers.add_parser('ingest-textbook-corpus')
    ingest_corpus.add_argument('--manifest', default='configs/textbook_sources.yaml')
    ingest_corpus.add_argument('--raw-dir', default='../data/raw/textbooks')
    ingest_corpus.add_argument('--output-dir', default='../data/interim/textbooks_v2')
    ingest_corpus.set_defaults(func=cmd_ingest_textbook_corpus)

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

    normalize_mit_xlsx_dir = subparsers.add_parser('normalize-mit-writing-xlsx-dir')
    normalize_mit_xlsx_dir.add_argument('input_dir')
    normalize_mit_xlsx_dir.add_argument('output_path')
    normalize_mit_xlsx_dir.add_argument('--reviews-output')
    normalize_mit_xlsx_dir.add_argument('--comments-output')
    normalize_mit_xlsx_dir.add_argument('--manifest-output')
    normalize_mit_xlsx_dir.add_argument('--source-id', default='mit_2026')
    normalize_mit_xlsx_dir.add_argument('--tournament', default='MIT Science Bowl 2026')
    normalize_mit_xlsx_dir.add_argument('--year', type=int, default=2026)
    normalize_mit_xlsx_dir.set_defaults(func=cmd_normalize_mit_writing_xlsx_dir)

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
    run_baseline_eval_cmd.add_argument("--style-questions")
    run_baseline_eval_cmd.add_argument("--reviews")
    run_baseline_eval_cmd.add_argument("--textbook-chunks")
    run_baseline_eval_cmd.add_argument("--max-items", type=int)
    run_baseline_eval_cmd.add_argument("--max-per-category", type=int)
    run_baseline_eval_cmd.add_argument("--exclude-category", action="append", default=["energy"])
    _add_model_override_args(run_baseline_eval_cmd)
    run_baseline_eval_cmd.set_defaults(func=cmd_run_baseline_eval)

    demo_generate = subparsers.add_parser("demo-generate")
    demo_generate.add_argument("--spec", required=True)
    demo_generate.add_argument("--textbook-chunks", required=True)
    demo_generate.add_argument("--style-questions", required=True)
    demo_generate.add_argument("--output-dir", required=True)
    _add_model_override_args(demo_generate)
    demo_generate.set_defaults(func=cmd_demo_generate)

    review_generated_cmd = subparsers.add_parser("review-generated-questions")
    review_generated_cmd.add_argument("runs_path")
    review_generated_cmd.add_argument("--output-path")
    review_generated_cmd.add_argument("--reviewer-id", default="local_reviewer")
    review_generated_cmd.add_argument("--host", default="127.0.0.1")
    review_generated_cmd.add_argument("--port", type=int, default=8775)
    review_generated_cmd.add_argument("--title", default="Generated Question Review")
    review_generated_cmd.set_defaults(func=cmd_review_generated_questions)

    generate_from_config_cmd = subparsers.add_parser("generate-from-config")
    generate_from_config_cmd.add_argument("config_path")
    _add_model_override_args(generate_from_config_cmd)
    generate_from_config_cmd.set_defaults(func=cmd_generate_from_config)

    export_sft_cmd = subparsers.add_parser("export-sft-dataset")
    export_sft_cmd.add_argument("--questions", required=True)
    export_sft_cmd.add_argument("--output-dir", required=True)
    export_sft_cmd.add_argument("--validation-fraction", type=float, default=0.05)
    export_sft_cmd.add_argument("--seed", type=int, default=42)
    export_sft_cmd.add_argument("--exclude-category", action="append", default=["energy"])
    export_sft_cmd.set_defaults(func=cmd_export_sft_dataset)

    clean_sft_cmd = subparsers.add_parser('build-clean-sft-dataset')
    clean_sft_cmd.add_argument('--questions', action='append', required=True)
    clean_sft_cmd.add_argument('--output-dir', required=True)
    clean_sft_cmd.add_argument('--review', action='append', default=[])
    clean_sft_cmd.add_argument('--profile', choices=CLEAN_SFT_PROFILES, default='mit_recent_plus_nsb')
    clean_sft_cmd.add_argument('--validation-fraction', type=float, default=0.05)
    clean_sft_cmd.add_argument('--test-fraction', type=float, default=0.05)
    clean_sft_cmd.add_argument('--seed', type=int, default=42)
    clean_sft_cmd.add_argument('--held-out-question-ids')
    clean_sft_cmd.add_argument('--minimum-writing-quality', type=float, default=0.0)
    clean_sft_cmd.add_argument('--include-unrated-writing', action='store_true')
    clean_sft_cmd.set_defaults(func=cmd_build_clean_sft_dataset)

    train_lora_cmd = subparsers.add_parser("train-lora-sft")
    train_lora_cmd.add_argument("config_path")
    train_lora_cmd.set_defaults(func=cmd_train_lora_sft)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
