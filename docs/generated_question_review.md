# Generated question review

Generate a batch from a config file:

```powershell
.venv\Scripts\python.exe -m scibowl.cli.main generate-from-config `
  configs\generation_job.example.yaml `
  --writer-model MODEL_NAME `
  --writer-base-url OPENAI_COMPATIBLE_URL `
  --verifier-model VERIFIER_MODEL_NAME `
  --verifier-base-url OPENAI_COMPATIBLE_URL `
  --disable-writer-fallback
```

The command writes one `GeneratedQuestionRunRecord` per draft. Each row contains the request, retrieved sources, draft, and verifier report.

Run the local review site against that JSONL file:

```powershell
.venv\Scripts\python.exe -m scibowl.cli.main review-generated-questions `
  ../data/processed/generated/example_batch.jsonl `
  --output-path ../data/processed/reviews/example_batch_reviews.jsonl `
  --reviewer-id brian
```

Then open `http://127.0.0.1:8775`.

For each draft, a reviewer can save:

- difficulty from 1 to 7
- quality as -1, 0, or 1
- a comment
- optional corrections to the automated format, factuality, topic, style, and overall scores

The output uses the generated draft ID as `question_id`. It contains only reviewed drafts, so review can stop and resume. Keep runs and reviews under `../data`, outside the repo.

Automated scores are triage signals. Compare models with blind human ratings of their generated drafts; do not transfer ratings from source questions to generated questions.
