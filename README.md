# scibowl-gpt

Science Bowl question-generation workspace with four main concerns:

- ingest textbooks, packet PDFs, and MIT writing data into normalized artifacts
- retrieve factual textbook chunks and style examples
- generate and verify draft questions through stable schemas
- evaluate prompting and future fine-tuning runs against held-out rated data

The current implementation is structured as a modular Python package under `src/scibowl` with typed schemas, CLI entrypoints, packet normalization/parsing tools, and baseline evaluation plumbing.

Key directories:

- `configs/` for runtime settings
- `src/scibowl/ingest/` for dataset, textbook, review, and packet ingestion
- `src/scibowl/generate/` and `src/scibowl/verify/` for draft production and review
- `src/scibowl/eval/` for splits and baseline evaluation
- `data/processed/` and `data/interim/` for normalized artifacts

Install locally with:

```powershell
pip install -e .[dev]
```

Run the CLI with:

```powershell
python -m scibowl.cli.main --help
```
