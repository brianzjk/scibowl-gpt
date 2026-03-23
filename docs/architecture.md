# Architecture

The project is centered on a deterministic artifact pipeline:

1. ingest raw textbooks and question sets into normalized records
2. retrieve fact chunks and style examples for a requested question spec
3. generate a draft question
4. verify the draft with structured checks
5. store final artifacts for evaluation or later fine-tuning

The current implementation uses local heuristics as placeholders for model-backed components so the package contracts are stable before production integrations are added.
