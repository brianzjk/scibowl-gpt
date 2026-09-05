# Tinker post-training

The clean SFT export already uses Tinker's conversation JSONL format. Each row has a
`messages` list; the extra ID and source fields let us audit the data and do not enter
the prompt.

Run Tinker from Linux, macOS, or WSL. Inkling's renderer does not support native
Windows at present.

## Before spending credits

Rebuild the data as described in `clean_data.md`, then run:

```bash
python -m scibowl.cli.main validate-tinker-sft \
  ../data/processed/training/clean_sft_v2
```

Do not start the main run until the held-out ID list is fixed and the manifest's
`held_out_excluded_question_ids` has been reviewed. The current split is enough for
an SFT trial, but it is not a final model benchmark.

Install the optional training tools only in the Linux training environment:

```bash
pip install -e '.[tinker]'
export TINKER_API_KEY='...'
```

## First experiments

Start with short capped runs on Inkling-Small. Sweep two or three learning rates,
keep the data and seed fixed, and compare validation loss plus blind human review.
Then repeat the best setting on Inkling. Do not spend credits on DPO or RL yet; this
repo has no sound preference or reward data for either method.

The runner uses Tinker's recommended renderer, trains only on assistant messages,
and keeps the curated validation split separate:

```bash
python scripts/train_tinker_sft.py \
  ../data/processed/training/clean_sft_v2 \
  ../data/runs/inkling-small-lr1 \
  --model thinkingmachines/Inkling-Small \
  --learning-rate 0.0001 \
  --max-steps 10
```

Remove `--max-steps` only after the capped run succeeds. Use a new log path for each
setting. Tinker charges by tokens, so check its live pricing page before a full run.

The `test` and `held_out` files never enter the training runner. Use them later for
blind generation and human ranking, not for checkpoint choice.
