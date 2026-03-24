# Repo Split Plan

This document describes how to split parts of `scibowl-gpt` into separate repositories later, after the question generator is working end to end.

## Current Recommendation

If only one subsystem gets extracted, it should be the duplicate / similarity tool.

Why:

- it is reusable outside the generator and training workflow
- it already has a clear input/output shape
- it is useful for future tournament writing even if the generator changes
- it is less tied to model orchestration than the rest of the repo

Do not split the scrapers first unless they become independently reusable and stable across multiple workflows.

## Suggested Future Repo Layout

Most likely long-term layout:

1. `scibowl-generator`
   Main repo for ingestion orchestration, retrieval, writer, verifier, evaluation, and training.

2. `scibowl-similarity`
   Standalone duplicate mining and review tool for normalized question corpora.

3. `scibowl-ingest` (optional, later)
   Packet downloaders, packet normalization, textbook ingestion, and tournament-specific parsing tools.

If `scibowl-ingest` is extracted, it should happen after the generator and similarity split are already stable.

## What Should Stay Here For Now

Keep these in the main repo:

- writer and verifier prompts
- model backends
- retrieval logic
- evaluation logic
- training code
- experiment configs

These parts are too tightly coupled to the question-generation system to split early without adding a lot of coordination overhead.

## Best First Extraction

The first extraction should be:

- `src/scibowl/dedupe/`
- duplicate-related schema
- duplicate review CLI commands
- duplicate review docs

That gives you a standalone workflow for:

- loading normalized question corpora
- mining similar pairs
- reviewing candidate duplicates
- exporting reviewed labels

## Minimal Portable Input Schema

Before splitting, define a small portable input format for the similarity repo.

It should be smaller than the full internal `NormalizedQuestion` schema and contain only what duplicate review actually needs:

```json
{
  "question_id": "mit_2026_rr1_12",
  "source_id": "mit_2026",
  "tournament": "MIT Science Bowl 2026",
  "round": "RR1",
  "category": "physics",
  "subcategory": "mechanics",
  "question_type": "tossup",
  "answer_mode": "short_answer",
  "question_text": "Question text here",
  "answer_text": "ANSWER: angular momentum",
  "choices": []
}
```

This is the key abstraction. Once the similarity repo can operate on this format, it no longer needs most of the generator repo.

## Similarity Repo Scope

`scibowl-similarity` should own:

- embedding-based candidate generation
- duplicate pair schemas
- reviewed pair schemas
- CSV export for manual review
- local review website
- label import/export
- future duplicate clustering, if added

It should not own:

- packet parsing
- textbook ingestion
- writer/verifier pipelines
- training logic
- model-serving orchestration

## Similarity Repo Inputs And Outputs

### Inputs

- normalized question JSONL in the portable schema
- optional reviewed duplicate JSONL for resuming sessions
- runtime config such as embedding model, threshold, top-k, and device

### Outputs

- candidate pair JSONL
- review CSV
- reviewed pair JSONL
- optional clustering outputs later

## What Needs To Be Abstracted Before Splitting

These are the main current couplings that should be reduced first.

### 1. Question schema coupling

Current duplicate review code depends on the main repo's question schema and source metadata conventions.

Needed change:

- create a minimal question schema specifically for similarity work
- add a converter in the main repo from `NormalizedQuestion` to that minimal schema

### 2. CLI coupling

Current dedupe commands are wired into:

- `src/scibowl/cli/main.py`

Needed change:

- duplicate review CLI should become its own entrypoint in the future repo
- the main repo should call it externally or just produce the input files it needs

### 3. Path and artifact naming conventions

The current workflow assumes this repo's `data/interim/duplicates` layout.

Needed change:

- treat paths as explicit CLI arguments
- document artifact conventions clearly
- avoid hard assumptions about sibling files or repo-local directories

### 4. Shared helper imports

The duplicate code currently depends on:

- schema exports
- JSONL helpers
- some repo-local utilities

Needed change:

- either copy in the minimal utilities the similarity repo needs
- or create a tiny shared package later if multiple repos truly need the same helpers

## File Mapping For A Future Similarity Split

Most of these would move directly:

- `src/scibowl/dedupe/candidates.py`
- `src/scibowl/dedupe/export.py`
- `src/scibowl/dedupe/review.py`
- `src/scibowl/dedupe/review_server.py`
- `src/scibowl/dedupe/review_app.html`
- `src/scibowl/schema/duplicate.py`
- dedupe-related tests
- duplicate review docs

These would likely be replaced, not copied directly:

- question schema imports from `src/scibowl/schema/question.py`
- CLI wiring from `src/scibowl/cli/main.py`

## Suggested Migration Order

### Phase 1. Freeze the interfaces inside this repo

Before extracting anything:

- define the portable question schema for similarity
- add an exporter from internal normalized questions to that portable schema
- make sure the review site works entirely off the portable schema

### Phase 2. Extract the similarity repo

- create new repo
- move dedupe code and tests
- add standalone packaging and CLI
- move duplicate review docs
- verify the review website still works with exported question files

### Phase 3. Reconnect the main repo

The main repo should then:

- export normalized questions for similarity review
- call the similarity tool manually or via a shell script
- import reviewed duplicate labels back into the training/filtering workflow

### Phase 4. Decide whether ingestion should also split

Only do this later, after the generator and similarity split have settled.

## Suggested Main Repo Responsibilities After The Split

The main repo would still own:

- all normalization pipelines
- dataset manifests
- writer/verifier generation flow
- evaluation and training
- import of reviewed duplicate labels into filtering and split logic

That keeps the training and generation system coherent while still letting the duplicate-review process be reusable across tournaments.

## Optional Future Ingest Repo

If you later split scrapers and ingestion, that repo should likely own:

- packet downloaders
- Word-to-PDF conversion helpers
- packet parsing
- textbook extraction and chunking
- tournament-specific parser rules

But that only makes sense once:

- the packet formats you care about are stable enough
- the normalized output schema is stable
- you actually want to reuse ingestion separately from generation

Right now, ingestion is still evolving too quickly for a split to be worth it.

## Risks Of Splitting Too Early

- schema drift between repos
- harder local testing
- duplicated config and utility code
- slower iteration when parser and generator changes need to move together
- more annoying setup for a project that is still changing quickly

## Concrete Pre-Split Checklist

Before extracting `scibowl-similarity`, make sure all of these are true:

- the question generator is running end to end
- the normalized question schema is stable
- duplicate review labels are stable enough to keep
- the portable similarity input schema exists
- the main repo can export that schema directly
- the similarity website works from exported files only
- there is a clear import path for reviewed labels back into training data filtering

## Difficulty Estimate

If done later, after the generator is stable:

- similarity repo extraction: medium, likely `1-3 days`
- ingest repo extraction: medium to high, likely `2-5 days` depending on how many parser exceptions exist at that point

The hard part is not moving files. The hard part is preserving clean contracts between repos.

## Recommended Decision

When you do this later:

1. split the similarity tool first
2. keep ingestion in the main repo for longer
3. only split ingestion after the normalized input/output contracts have settled
