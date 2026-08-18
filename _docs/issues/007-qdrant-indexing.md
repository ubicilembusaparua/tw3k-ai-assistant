# TASK-007 — Add idempotent Qdrant initialization and reindexing

Status: Groomed

## Goal

Compose populates an empty `tw3k_transcripts` collection from the tracked
dataset and leaves an existing populated collection untouched on normal
startup. Operators have an explicit, visibly destructive force-reindex path
for dataset changes.

## Acceptance criteria

- [ ] `ingest_qdrant.py` reads `QDRANT_URL`, `QDRANT_COLLECTION`,
      `QDRANT_BATCH_SIZE`, and `QDRANT_FORCE_REINDEX` from the environment,
      with collection `tw3k_transcripts`, batch size `64`, and force reindex
      `false` as defaults.
- [ ] Boolean parsing accepts the documented true/false forms and rejects an
      invalid value; a non-positive or invalid batch size exits non-zero before
      changing the collection.
- [ ] With force reindex disabled, a collection containing points is not
      deleted, re-embedded, or upserted; the command reports that indexing was
      skipped and exits 0.
- [ ] A missing or zero-point collection is populated from every indexable
      chunk in `tw3k_dataset.jsonl`, using the configured batch size, and the
      command verifies or reports the resulting point count.
- [ ] With force reindex enabled, the target collection is explicitly
      recreated and populated from the current dataset; the command clearly
      reports the destructive mode and exits non-zero if any step fails.
- [ ] An empty or unreadable dataset cannot be reported as a successful,
      ready-to-serve index; the failure explains what is missing.
- [ ] The one-shot `qdrant-init` service waits for successful `model-init` and
      a healthy Qdrant service, uses the internal Qdrant URL and shared model
      volume, and returns a failure status when ingestion fails.
- [ ] A documented command can run `qdrant-init` with
      `QDRANT_FORCE_REINDEX=true` without changing the normal startup default.

## Out of scope

- [ ] Qdrant server health/storage and application connection fallback,
      moved to [TASK-005](005-qdrant-connectivity.md).
- [ ] Model download implementation and the final app volume wiring, moved
      to [TASK-006](006-model-preparation.md) and
      [TASK-008](008-compose-application.md).
- [ ] General operator documentation, deterministic tests, and complete
      lifecycle evidence, moved to
      [TASK-009](009-operator-documentation.md),
      [TASK-010](010-container-tests.md), and
      [TASK-011](011-compose-lifecycle.md).

## Constraints

- Normal startup must default to non-destructive behavior.
- Use the tracked `tw3k_dataset.jsonl` and existing ingestion/retriever code;
  do not introduce a second indexing implementation or dependency.
- Run ingestion against the configured service URL, never an implicit
  `localhost` or in-memory client in Compose.
- Keep the force option explicit in both configuration and operator-facing
  output because it replaces stored vectors.
