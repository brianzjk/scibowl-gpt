# Textbook search

The old textbook corpus should not be reused. Its chunks often had 400 to 650 words, lacked page data, and included front matter, answers, glossaries, or credits.

`configs/textbook_sources.yaml` lists exact content pages for the 11 current PDFs. Each entry also stores the PDF page count and SHA-256 hash. A changed PDF fails fast instead of using stale page ranges.

Build a new, separate corpus with:

```powershell
python -m scibowl.cli.main ingest-textbook-corpus `
  --manifest configs/textbook_sources.yaml `
  --raw-dir ../data/raw/textbooks `
  --output-dir ../data/interim/textbooks_v2
```

The builder writes chunks of at most 180 words with 30 words of overlap. Overlap can cross a PDF page, and each chunk records every physical page it spans. Stable IDs use the document, page span, and content hash. The metadata also records the source hash, corpus version, and one shared corpus run ID.

The cleaner drops pages that PDF extraction identifies as chapter-review or selected-answer pages. Ahrens chapter-opening contents are stripped while the prose after them is kept. This costs a small amount of summary text but prevents source questions from leaking into retrieval.

Automatic front and back trimming remains a fallback for new books. For a fixed training corpus, audit the PDF and add exact bounds to the manifest. Page numbers are reliable. Chapter and section labels use a strict parser and may be absent when the text is unclear.

Search now:

- filters to the category book list
- treats Earth and Space subcategory routes as a small preference, not a hard one-book rule
- ranks facts with weighted BM25
- requires a direct topic or subcategory match and returns no facts on a miss
- removes duplicate and recently used chunk IDs
- uses a stable tie order that does not depend on the request ID
- returns title, chapter or section when safe, and PDF pages as the locator
- asks the writer to name only chunks that directly support its question

Before trusting search quality, build a small fixed set of about 50 queries across the main categories plus several no-answer cases. Mark the supporting book and PDF pages by hand. Track page recall at 3, mean reciprocal rank, no-answer accuracy, duplicate rate, bundle word count, and latency. Run the same set for each routing, scoring, and chunk-size ablation.
