# Clean training data

Use `build-clean-sft-dataset` for new SFT work. The older `export-sft-dataset` command remains for repeat runs of the first baseline, but it uses guessed labels and old sample weights. Do not use that export to judge a new model.

The clean export keeps these rules:

- category and question type always come from the source record
- difficulty comes only from human reviews or an explicit source label
- missing difficulty and subcategory stay missing and do not appear in the training prompt
- every example has weight 1; there is no late-round NSB or blanket MIT bonus
- exact copies collapse to one record
- near copies share one split, and a held-out ID moves its whole copy group out of training
- current MIT Energy means a real current MIT research setup followed by a related science or engineering question
- old or official NSB Energy is excluded because it does not share that meaning
- MIT writing-sheet questions need a mean quality rating of at least 0
- unrated writing-sheet drafts stay in the normalized corpus but stay out of SFT by default
- official MIT packets and official NSB remain eligible without ratings because tournament use already vetted them

Import the 2026 Google Sheets exports before building training data:

```powershell
python -m scibowl.cli.main normalize-mit-writing-xlsx-dir `
  ../data/raw/question_sets/mit_2026 `
  ../data/interim/question_sets/mit_2026.jsonl `
  --reviews-output ../data/processed/reviews/mit_2026.jsonl `
  --comments-output ../data/interim/question_sets/mit_2026_comments.jsonl `
  --manifest-output ../data/interim/question_sets/mit_2026_manifest.json
```

This import reads the `Template` tab, skips visual bonuses and incomplete rows, and computes labels from the individual rating cells. It ignores difficulty values outside 1–7 and quality values outside -1–1. Threaded Google comments and plain notes go to a sidecar; they never enter prompts or targets.

The default profile includes MIT 2024 and later plus official NSB. It keeps official NSB as useful supporting style data even though those records lack human ratings.

```powershell
python -m scibowl.cli.main build-clean-sft-dataset `
  --questions ../data/interim/question_sets/style_corpus_all.jsonl `
  --questions ../data/interim/question_sets/mit_2026.jsonl `
  --review ../data/processed/reviews/mit_2025.jsonl `
  --review ../data/processed/reviews/mit_2026.jsonl `
  --profile mit_recent_plus_nsb `
  --held-out-question-ids configs/held_out_question_ids.txt `
  --output-dir ../data/processed/training/clean_sft_v2
```

If there is no held-out ID file yet, leave out that option. Add the fixed evaluation IDs before any final training run.

The manifest reports how many requested holdout IDs were present and how many survived curation. Review `held_out_excluded_question_ids` before freezing an evaluation set; do not infer benchmark coverage from the held-out split count alone.

Use `--include-unrated-writing` only for an explicit ablation. Do not use it for the main SFT run.

For ablations, rerun with `mit_recent`, `mit_all`, `mit_recent_plus_nsb`, and `mit_all_plus_nsb`. Keep the same seed and held-out file. The output manifest records the profile, source mix, label counts, exclusions, and split counts.

The output has `train`, `val`, `test`, and `held_out` JSONL files. Do not tune prompts or checkpoints on `test` or `held_out`.
