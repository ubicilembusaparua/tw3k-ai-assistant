# TASK-011 — Validate the complete Compose lifecycle

Status: Groomed

## Goal

A clean checkout can build and start the complete Compose stack, serve the
Streamlit application after all initialization jobs succeed, reuse persisted
state on restart, and perform an explicit vector rebuild without hidden manual
steps.

## Acceptance criteria

- [ ] In an isolated checkout/environment, `docker compose config` exits 0 and
      `docker compose build` succeeds with no missing files, unresolved
      required variables, or secret leakage in the build output.
- [ ] Starting from removed test volumes with `docker compose up --build`
      results in successful `model-init`, `db-init`, and `qdrant-init` jobs,
      creates the named volumes, and leaves `app`, `postgres`, and `qdrant`
      running.
- [ ] The first-run checks show the model files in the model volume, both
      database tables present and empty, a non-zero Qdrant point count matching
      the indexed dataset, and an HTTP response from Streamlit on the
      configured host port.
- [ ] After `docker compose down` followed by `docker compose up`, the model
      is reused, Qdrant indexing is skipped for the populated collection, and
      persistent database/Qdrant state remains unchanged.
- [ ] Running the documented force-reindex command replaces the Qdrant
      vectors successfully, while a normal subsequent startup returns to the
      skip behavior.
- [ ] If `db-init`, `model-init`, or `qdrant-init` is made to fail in the
      isolated test environment, the app does not start as if initialization
      succeeded.
- [ ] `docker compose down` preserves the named application volumes, while
      `docker compose down -v` removes those test volumes and their model,
      database, and Qdrant data.
- [ ] `uv run pytest` passes after the Compose validation.

## Out of scope

- [ ] Implementing any behavior under test; those deliverables belong to
      [TASK-002](002-reproducible-image.md) through
      [TASK-010](010-container-tests.md) and must satisfy their own acceptance
      criteria first.

## Constraints

- Use a disposable checkout or explicitly disposable named volumes for any
  destructive reset; never remove an operator's existing application data.
- Run the lifecycle with a configured but non-production credential and do not
  submit secrets or full service logs containing secrets as evidence.
- Verify both fresh and persisted paths; a single successful `up` is not
  sufficient evidence for this issue.
- Keep the validation independent of the ignored `dataset_builder` directory
  and use only the tracked dataset.
