# Generated Question Review

The baseline evaluation now writes two files:

- `*_baseline_eval.jsonl`
  Summary scores only.
- `*_baseline_runs.jsonl`
  Full generated-question artifacts for review:
  - spec
  - generated draft
  - verifier report
  - evaluation scores

For the current Ollama baseline, the main review corpus is:

- Iteration-sized sample, recommended for prompt tuning:
  - `data/processed/eval/baseline_ollama_gptoss20b_iter_5_per_category_v2/test_baseline_runs.jsonl`
  - `25` questions total, `5` each from biology, chemistry, earth/space, math, and physics
  - `energy` is intentionally excluded
- Full benchmark run:
  - `data/processed/eval/baseline_ollama_gptoss20b/test_baseline_runs.jsonl`
  - `189` questions total

## Run The Review Website

From the repo root:

```powershell
.venv\Scripts\python.exe -m scibowl.cli.main review-generated-questions `
  data/processed/eval/baseline_ollama_gptoss20b_iter_5_per_category_v2/test_baseline_runs.jsonl `
  --output-path data/processed/reviews/baseline_ollama_gptoss20b_iter_5_per_category_v2_reviews.jsonl `
  --reviewer-id brian `
  --host 127.0.0.1 `
  --port 8775 `
  --title "Generated Question Review"
```

Then open:

`http://127.0.0.1:8775`

## What The Site Shows

For each generated question, the website shows:

- the generated question, answer, and multiple-choice choices if present
- the requested spec: category, subcategory, type, answer mode, difficulty, topic focus
- the writer and verifier model names
- verifier summary plus the current verifier rubric scores: format/grammar, on-topic, style, factuality, and overall
- blankable human override scores for those same verifier metrics

## What You Can Save

Each reviewed generated question stores:

- `difficulty`: integer `1-7`
- `quality`: one of `-1`, `0`, `1`
- `comment`: optional free text
- `human_verifier_scores`: optional per-metric overrides in the range `0.0` to `1.0`
- `verifier_comment`: optional free text about verifier mistakes

Reviews are saved incrementally to the JSONL you pass as `--output-path`. That file is sparse: only reviewed questions are written, so you can stop and resume safely.

## Notes

- The review file uses the generated draft id as `question_id`, not the original MIT question id.
- `Prev` / `Next` let you navigate the queue.
- `Unreviewed`, `All`, and `Reviewed` filters are available in the UI.
- The website recalculates the displayed verifier scores from the stored verifier report, so older generation runs can still be reviewed under the current rubric without rerunning generation.
- If you rerun baseline generation into the same output directory, `test_baseline_runs.jsonl` will resume missing artifacts instead of duplicating existing run rows.
- For faster iteration, prefer the `iter_5_per_category_v2` run over the full benchmark run.
