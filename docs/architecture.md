# Architecture

The project uses a file-based pipeline:

1. ingest audited textbook PDFs and normalized question sets
2. build clean, source-aware training splits
3. retrieve factual chunks and separate style examples for a question request
4. ask a writer model for one draft and the chunk IDs that support it
5. run rule and model checks as triage, not as the final score
6. send every candidate to a human before tournament use

The fact corpus uses fixed PDF page ranges, source hashes, stable chunk IDs, and physical page locators. Search uses weighted BM25 with category filters, soft Earth and Space book preferences, no-match abstention, and diverse top results.

The clean SFT export keeps missing labels missing, groups near copies into one split,
and quarantines every eligible question from the fixed MIT 2025 holdout list. Recent
MIT and official NSB are separate source families so their mix can be tested. The
2026 workbook importer preserves individual ratings and Google comments in sidecars;
comments never enter training examples. Unrated or negatively rated writing drafts
stay out of the main SFT export.

Writer and verifier clients use OpenAI-compatible endpoints when configured. Local heuristics remain only as a development fallback. Model comparison requires blind human ratings of the generated drafts; ratings attached to source questions cannot score new drafts.
