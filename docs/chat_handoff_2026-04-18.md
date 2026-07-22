# Scibowl Codex Handoff (2026-04-18)

This file summarizes the important context from the current long Codex CLI session so a new Codex app chat can pick up the work with minimal loss of context.

## Workspace

- Main repo: `C:\Users\brian\Documents\Scibowl\scibowl-gpt`
- Shared data directory: `C:\Users\brian\Documents\Scibowl\data`
- Sibling repos:
  - `C:\Users\brian\Documents\Scibowl\scibowl-similarity`
  - `C:\Users\brian\Documents\Scibowl\scibowl-packet-scraper`

## Repo split

The old monorepo was split into:

- `scibowl-gpt`
  - generation
  - verification
  - evaluation
  - textbook/question-set/review ingestion
  - generated-question review site
- `scibowl-similarity`
  - duplicate mining
  - duplicate review website
- `scibowl-packet-scraper`
  - packet downloaders
  - filename normalization
  - packet parsing

Shared artifacts now live under `C:\Users\brian\Documents\Scibowl\data`, outside the repo.

## Data / corpora status

- Style corpus path:
  - `C:\Users\brian\Documents\Scibowl\data\interim\question_sets\style_corpus_all.jsonl`
- Packet/textbook/generated data also lives under the shared `data` directory.
- MIT 2022 and MIT 2024 were added and parsed.

## Current generation pipeline

### Writer / verifier setup

- The project supports local Ollama inference via the OpenAI-compatible path.
- `gpt-oss:20b` is the preferred local writer baseline.
- `qwen3.5` was tested but was not a good fit because it produced long reasoning traces and timed out in the current setup.
- The generation CLI/script supports:
  - `--ollama-model`
  - `--writer-model`
  - `--writer-base-url`
  - `--verifier-model`
  - `--verifier-base-url`
  - `--writer-timeout-seconds`
  - `--verifier-timeout-seconds`
  - `--disable-writer-fallback`

### Important writer behavior changes

- `answer_mode` is now writer-selected, not fixed ahead of time.
- The stored `answer_mode` is inferred from the generated output.
- Writer fallback can be disabled; if disabled, generation fails loudly instead of silently writing heuristic placeholder drafts.
- The writer now retries malformed / unusable outputs from `gpt-oss:20b`.

Relevant files:

- `src/scibowl/generate/local_model.py`
- `src/scibowl/llm/client.py`
- `src/scibowl/llm/runtime.py`
- `scripts/generate_questions.py`
- `src/scibowl/cli/main.py`

## Generated-question review site

The generated-question review site exists and supports:

- rating difficulty `1-7`
- rating quality `-1 / 0 / 1`
- comments
- verifier override scores / notes

Run it with:

```powershell
.venv\Scripts\python.exe -m scibowl.cli.main review-generated-questions `
  ..\data\processed\generated\<runs>.jsonl `
  --output-path ..\data\processed\reviews\<reviews>.jsonl `
  --reviewer-id brian `
  --host 127.0.0.1 `
  --port 8775 `
  --title "Generated Question Review"
```

Then open `http://127.0.0.1:8775`.

## ESS retrieval / chunking work

### Routing

ESS retrieval is routed by subcategory:

- `Cosmology`, `Solar System`, `Stars` -> `Seeds`
- `Observation` -> `Burns`
- `Meteorology` -> `Ahrens`
- `Hydrology` -> `Garrison`
- `Rocks and Minerals`, `Tectonics` -> `Tarbuck`

Relevant file:

- `src/scibowl/retrieval/search.py`

### Diversity pressure

Generation batch retrieval now avoids recently used fact chunk IDs and style question IDs so the same bundles are less likely to repeat immediately.

Relevant files:

- `src/scibowl/generate/orchestration.py`
- `src/scibowl/generate/batch.py`

### Textbook chunking changes

Chunking was improved to:

- trim some front matter and back matter
- strip many exercise / review artifacts such as:
  - `Problem ...`
  - `Concept Check`
  - `Key Terms`
  - review-question blocks
- chunk by paragraph rather than pure sliding token windows

Relevant files:

- `src/scibowl/ingest/textbooks.py`
- `src/scibowl/utils/text.py`

### Important remaining issue

Chunking is still not clean enough, especially for astronomy and some earth-science books.

Concrete findings from the current data:

- `seeds_foundations_of_astrophysics_chunks.jsonl` still begins with front matter like:
  - `A Note to the Student`
  - edition-change / acknowledgements material
- `garrison_essentials_of_oceanography_5e_chunks.jsonl` still starts with TOC/preface-like material
- `ahrens_essentials_of_meteorology_chunks.jsonl` is cleaner, but still contains broad intro material

Also, repeated ESS questions are still often caused by retrieval narrowing too aggressively to a small chunk neighborhood.

In the latest 40-question earth/space run:

- `Cosmology`: 5 questions, only 3 unique retrieval bundles
- all cosmology questions drew from a narrow Seeds chunk band around `0386-0407`
- repeated chunk reuse is still a problem

Conclusion:

- automated chunking is not enough by itself
- the next strong fix is likely semi-manual section/chapter manifests per subcategory, especially for `Cosmology`

## Latest earth/space generation run

Latest completed run:

- file:
  - `C:\Users\brian\Documents\Scibowl\data\processed\generated\earth_20_space_20_gptoss20b.jsonl`
- count: `40`
- split: `20` earth + `20` space
- writer models:
  - all `gpt-oss:20b`
- verifier models:
  - all `rule-verifier-v1`

Subcategory spread:

- earth:
  - `Hydrology 8`
  - `Meteorology 7`
  - `Tectonics 3`
  - `Rocks and Minerals 2`
- space:
  - `Stars 8`
  - `Solar System 5`
  - `Cosmology 5`
  - `Observation 2`

Generation command used:

```powershell
$env:SCIBOWL_WRITER_RETRIES='6'
.venv\Scripts\python.exe scripts\generate_questions.py `
  configs\generation_earth_20_space_20_gptoss20b.yaml `
  --writer-model gpt-oss:20b `
  --writer-base-url http://127.0.0.1:11434/v1 `
  --writer-timeout-seconds 600 `
  --disable-writer-fallback
```

Relevant config:

- `configs/generation_earth_20_space_20_gptoss20b.yaml`

## SFT / training status

### Dataset export

There is a first SFT export pipeline in `scibowl-gpt`.

Relevant files:

- `src/scibowl/training/sft.py`
- `src/scibowl/schema/training.py`
- `docs/sft_export.md`

Current exported dataset location:

- `C:\Users\brian\Documents\Scibowl\data\processed\training\sft_v1`

SFT export design:

- goal: teach Science Bowl format / conventions / category fit
- not meant to solve final quality by itself
- MIT questions are upweighted
- official NSB rounds `> 11` are upweighted
- `energy` excluded
- malformed / contaminated rows excluded

### LoRA training

LoRA training support was added.

Relevant files:

- `src/scibowl/training/lora.py`
- `scripts/train_lora_sft.py`
- `docs/lora_training.md`
- configs:
  - `configs/lora_sft.example.yaml`
  - `configs/lora_sft.gptoss20b.example.yaml`

Important recent training notes:

- A zero-loss bug was fixed by switching SFT to a compact training prompt so the assistant completion is not truncated away.
- Qwen was explored for SFT, but GPT-OSS is now the preferred path.
- Original Qwen configs were kept in case a different model is tried later.

## Model-specific observations

### GPT-OSS

- `gpt-oss:20b` works well enough for local writing, but sometimes emits malformed or unusable JSON.
- Common bad payloads seen:
  - empty content
  - payloads with only `process`
  - payloads with irrelevant JSON keys
  - malformed `choices`
- The writer retry / parsing hardening was added to survive these cases.

### Qwen 3.5

- Qwen 3.5 on Ollama produced overly long reasoning traces and timed out for full writer prompts.
- It was not a good fit for the current local generation path.

## Duplicate review / similarity

That work now lives in the sibling repo:

- `C:\Users\brian\Documents\Scibowl\scibowl-similarity`

This repo contains:

- duplicate mining
- candidate export
- duplicate review website

The main `scibowl-gpt` repo no longer owns that subsystem.

## Known operational issues

- PowerShell profile loading still throws execution-policy / OneDrive warnings in shell output. Those warnings are noisy but not core to the repo logic.
- The Codex alias/profile setup exists, but PowerShell execution policy still prevented normal profile loading in previous sessions.

## What a new chat should probably focus on next

Most useful next steps:

1. Clean up textbook chunking further for astronomy / Garrison / Tarbuck.
2. Add semi-manual chapter/section manifests per ESS subcategory, especially `Cosmology`.
3. Improve retrieval diversity further if repeated bundles are still dominating.
4. Review the latest 40 earth/space generations and decide whether chunking or prompting is the bigger remaining problem.
5. Continue SFT work once the corpus quality is stable enough.

## Suggested opening prompt for the next Codex app chat

You can start a new chat with something close to:

> Read `docs/chat_handoff_2026-04-18.md` first. This repo is `scibowl-gpt` with shared data in `C:\\Users\\brian\\Documents\\Scibowl\\data`. The current focus is earth/space generation quality, especially chunking/retrieval problems causing repetitive cosmology questions. Use the latest run at `..\\data\\processed\\generated\\earth_20_space_20_gptoss20b.jsonl` as the working example, and preserve the current GPT-OSS writer setup.
