# Pre-RAG baseline

Recorded on 2026-08-15 before RAG implementation.

- Active application directory: repository root
- Legacy dataset-builder directory: `dataset_builder/` (excluded from RAG work)
- Source snapshot: commit `87ec46a`
- Command: `uv run pytest`
- Platform: Windows, Python 3.13.14
- Result: 6 passed in 8.53 seconds

The baseline was run from an isolated archive of the source snapshot because the
working tree already contained an in-progress relocation of the legacy dataset
builder. This avoided reading from or modifying the excluded `dataset_builder/`
directory.
