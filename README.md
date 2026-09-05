# scibowl-gpt

Science Bowl question-generation workspace focused on:

- ingesting textbook, question-set, and review data into normalized artifacts
- retrieving factual textbook chunks and style examples
- generating and verifying draft questions through stable schemas
- collecting blind human reviews of generated questions

This repo owns data curation, generation, automated checks, and human review. It does not treat automated verifier scores as a substitute for model evaluation.

Related sibling repos:

- `../scibowl-similarity` for duplicate mining and the duplicate review website
- `../scibowl-packet-scraper` for packet downloading, filename normalization, and packet PDF parsing

Shared artifacts now live outside the repo in:

- `../data`

Key directories:

- `configs/` for runtime settings
- `src/scibowl/ingest/` for textbook, question-set, and review ingestion
- `src/scibowl/generate/` and `src/scibowl/verify/` for draft production and review
- `src/scibowl/generation_review/` for the local generated-question review website

Docs:

- `docs/architecture.md` for the high-level pipeline
- `docs/generation_config.md` for config-driven question generation
- `docs/generated_question_review.md` for reviewing generated questions locally
- `docs/clean_data.md` for the curated MIT and official NSB training export
- `docs/tinker_posttraining.md` for validating and running Inkling SFT on Tinker
- `docs/textbook_retrieval.md` for rebuilding and testing textbook search

Install locally with:

```powershell
pip install -e .[dev]
```

Run the CLI with:

```powershell
python -m scibowl.cli.main --help
```
