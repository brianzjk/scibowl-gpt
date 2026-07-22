# SFT Export

> Legacy baseline only. Do not use this export for a new Inkling or comparison run. It carries guessed difficulty and subcategory values, arbitrary source weights, and per-ID splits that can leak near copies. Use `docs/clean_data.md` and `build-clean-sft-dataset` instead.

The first SFT dataset in `scibowl-gpt` is meant to teach:

- Science Bowl formatting
- category fit
- subcategory fit when that metadata is available
- clue-flow and answer-line conventions

It is not meant to directly optimize overall question quality. That is a better fit for later RLHF or preference tuning.

## What The Exporter Does

Input:

- a normalized question corpus, usually `../data/interim/question_sets/style_corpus_all.jsonl`

Filters:

- excludes `energy` by default
- excludes malformed rows with missing question/answer text
- excludes malformed short-answer rows that still carry choices
- excludes malformed multiple-choice rows unless they have exactly four `W/X/Y/Z` choices and the stem contains `Which of the following`

Dedupe:

- removes exact duplicates based on normalized `question_text + answer_text + choices`
- keeps the higher-priority version when duplicates exist
- preference order favors higher-weight examples, dataset-backed rows, and rows with explicit quality metadata

Weights:

- default examples: `1.0`
- any `mit_*` source: `2.0`
- official `nsb_set_*` examples from rounds `> 11`: `1.5`

The canonical dataset stores `sample_weight` in metadata. A separate materialized train file approximates those weights by repeating rows deterministically.

## Output Files

The exporter writes these files under the chosen output directory:

- `sft_examples_all.jsonl`
- `sft_examples_train.jsonl`
- `sft_examples_val.jsonl`
- `sft_examples_train_materialized.jsonl`
- `sft_summary.json`

Each JSONL row looks like:

```json
{
  "example_id": "sft_mit_2025_0001",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{\"question_text\":\"...\",\"answer_text\":\"ANSWER: ...\",\"choices\":[]}"}
  ],
  "metadata": {
    "source_question_id": "mit_2025_0001",
    "source_dataset": "mit_2025",
    "source_tournament": "MIT Science Bowl 2025",
    "source_round_number": null,
    "category": "biology",
    "subcategory": "Cell Biology",
    "question_type": "tossup",
    "target_answer_mode": "multiple_choice",
    "difficulty": 1,
    "sample_weight": 2.0,
    "weight_reasons": ["mit"],
    "split": "train",
    "repeat_count": 2,
    "has_specific_subcategory": true,
    "dedupe_fingerprint": "..."
  }
}
```

Notes:

- the SFT exporter uses a compact training prompt rather than the full inference-time writer prompt
- `answer_mode` is not fixed in the input prompt; the model chooses short-answer vs multiple-choice from the training examples
- the assistant response is always JSON with `question_text`, `answer_text`, and `choices`

## Run It

From the repo root:

```powershell
.venv\Scripts\python.exe -m scibowl.cli.main export-sft-dataset `
  --questions ..\data\interim\question_sets\style_corpus_all.jsonl `
  --output-dir ..\data\processed\training\sft_v1
```

Optional flags:

- `--validation-fraction 0.05`
- `--seed 42`
- `--exclude-category energy`

## Which File To Train On

For a first LoRA run, use:

- `sft_examples_train_materialized.jsonl` for training
- `sft_examples_val.jsonl` for validation

Keep `sft_examples_all.jsonl` as the canonical unique dataset for inspection and later re-weighting.
