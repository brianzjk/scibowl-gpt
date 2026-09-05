# scibowl-gpt

Tools for writing MIT Science Bowl questions with language models. The project
curates training data, searches textbooks for factual support, generates drafts,
runs automated checks, and collects blind human reviews.

Humans must approve every question before tournament use. Automated checks help
reviewers find problems; they do not measure overall model quality.

## Current state

- Textbook search uses an audited corpus with fixed page ranges, source hashes,
  stable chunk IDs, and category-aware retrieval.
- The clean SFT builder keeps recent MIT questions and official National Science
  Bowl questions, removes unusable drafts and duplicates, and protects fixed data
  splits.
- The current clean export has 5,088 training, 269 validation, 287 test, and 140
  held-out examples.
- The Tinker runner is ready for capped Inkling and Inkling-Small SFT trials from
  Linux, macOS, or WSL.
- A final model comparison still needs blind human ratings of generated questions.

Data files live in the sibling `../data` directory rather than in Git. The related
`../scibowl-similarity` and `../scibowl-packet-scraper` repos handle duplicate
review and packet scraping.

## Setup and checks

```powershell
pip install -e .[dev]
python -m pytest
python -m scibowl.cli.main --help
python -m scibowl.cli.main validate-tinker-sft `
  ../data/processed/training/clean_sft_v2
```

Install `.[tinker]` only in the Linux, macOS, or WSL environment used for Tinker.

## Main paths

- `configs/` — retrieval, textbook, holdout, and generation settings
- `src/scibowl/ingest/` — question, rating, workbook, and textbook import
- `src/scibowl/retrieval/` — factual and style retrieval
- `src/scibowl/generate/` and `src/scibowl/verify/` — generation and checks
- `src/scibowl/training/` — clean SFT curation and preflight validation
- `src/scibowl/generation_review/` — local blind-review site

## Guides

- [Architecture](docs/architecture.md)
- [Clean training data](docs/clean_data.md)
- [Tinker post-training](docs/tinker_posttraining.md)
- [Textbook retrieval](docs/textbook_retrieval.md)
- [Generation jobs](docs/generation_config.md)
- [Human review](docs/generated_question_review.md)
