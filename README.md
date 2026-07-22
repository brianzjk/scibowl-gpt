# scibowl-gpt

Science Bowl question-generation workspace focused on:

- ingesting textbook, question-set, and review data into normalized artifacts
- retrieving factual textbook chunks and style examples
- generating and verifying draft questions through stable schemas
- evaluating prompting and future fine-tuning runs against held-out rated data

This repo now owns generation, verification, evaluation, and the generated-question review workflow.

Related sibling repos:

- `../scibowl-similarity` for duplicate mining and the duplicate review website
- `../scibowl-packet-scraper` for packet downloading, filename normalization, and packet PDF parsing

Shared artifacts now live outside the repo in:

- `../data`

Key directories:

- `configs/` for runtime settings
- `src/scibowl/ingest/` for textbook, question-set, and review ingestion
- `src/scibowl/generate/` and `src/scibowl/verify/` for draft production and review
- `src/scibowl/eval/` for splits and baseline evaluation
- `src/scibowl/generation_review/` for the local generated-question review website

Docs:

- `docs/architecture.md` for the high-level pipeline
- `docs/generation_config.md` for config-driven question generation
- `docs/generated_question_review.md` for reviewing generated baseline questions locally
- `docs/sft_export.md` for exporting the first supervised fine-tuning dataset
- `docs/clean_data.md` for the curated MIT and official NSB training export
- `docs/textbook_retrieval.md` for rebuilding and testing textbook search
- `docs/lora_training.md` for running the first local LoRA SFT job

Install locally with:

```powershell
pip install -e .[dev]
```

Run the CLI with:

```powershell
python -m scibowl.cli.main --help
```
