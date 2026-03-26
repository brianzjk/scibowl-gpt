# Config-Driven Question Generation

Use the config-driven generator to request new questions from a model without tying them to a held-out MIT question.

The example config is:

- `configs/generation_job.example.yaml`

You can run it either through the helper script:

```powershell
.venv\Scripts\python.exe scripts\generate_questions.py configs\generation_job.example.yaml
```

or through the CLI:

```powershell
.venv\Scripts\python.exe -m scibowl.cli.main generate-from-config configs\generation_job.example.yaml
```

The output is a JSONL of `GeneratedQuestionRunRecord` rows. Each row includes:

- the requested spec
- the retrieval bundle
- the generated draft
- the verifier report
- the draft-level evaluation scores

Those JSONL files can be reviewed in the generated-question review website with:

```powershell
.venv\Scripts\python.exe -m scibowl.cli.main review-generated-questions `
  ../data/processed/generated/example_batch.jsonl `
  --output-path ../data/processed/reviews/example_batch_reviews.jsonl `
  --reviewer-id brian `
  --host 127.0.0.1 `
  --port 8775 `
  --title "Generated Question Review"
```

## Config Shape

```yaml
output_path: ../data/processed/generated/example_batch.jsonl
style_questions_path: ../data/interim/question_sets/style_corpus_all.jsonl
textbook_chunks_path: ../data/interim/textbooks
random_seed: 7

jobs:
  - job_id: biology_membranes
    category: biology
    subcategory: cell biology
    question_type: tossup
    difficulty: 4
    count: 3
    topic_focus:
      - membrane transport
      - ion channels

  - job_id: ess_random_batch
    category: earth_space
    subcategory_mode: random
    question_type_mode: random
    difficulty_mode: random
    difficulty_pool: [3, 4, 5]
    count: 5
```

## Notes

- `energy` jobs are rejected.
- `subcategory` is required only for normal fixed jobs. If you set `subcategory_mode: random`, the generator will sample a subcategory for each requested question.
- `question_type` and `difficulty` also support `*_mode: random`.
- `answer_mode` is always writer-selected and is inferred from the model output.
- Built-in random pools:
  - `question_type`: `tossup`, `bonus`
  - `difficulty`: `1-7`
- You can optionally set `subcategory_pool` on a random job to override the default pool.
- You can also override `question_type_pool` or `difficulty_pool` on a random job.
- `random_seed` is optional, but useful if you want reproducible random sampling across all random job fields.
- The built-in random pools currently use these canonical subcategories:
  - `earth_space`: `Cosmology`, `Hydrology`, `Meteorology`, `Observation`, `Rocks and Minerals`, `Solar System`, `Stars`, `Tectonics`
  - `math`: `Algebra`, `Calculus`, `Combinatorics`, `Geometry`, `Number Theory`
- The random pools intentionally exclude broad/noisy labels like `other` and `Random`.
- The generator uses textbooks as factual reference material and style examples only for tone/format.
- The review website shows only the generated question and verifier output; it does not compare against a held-out reference question.
- After the repo split, the recommended convention is to keep generated artifacts under `../data` rather than inside the repo.
