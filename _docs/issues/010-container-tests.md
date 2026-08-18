# TASK-010 — Add tests for container configuration behavior

Status: Groomed

## Goal

Fast deterministic tests protect the environment parsing, connection-failure,
initialization, indexing, and model-cache contracts without requiring Docker,
external services, model downloads, or real credentials.

## Acceptance criteria

- [ ] Tests cover default and overridden `QDRANT_URL` and
      `QDRANT_COLLECTION` values, and prove that the configured values reach
      the retrieval client.
- [ ] Tests cover the default false value and an explicit true value for
      `QDRANT_FORCE_REINDEX`, including the non-force skip behavior for a
      populated collection.
- [ ] A test proves that an unavailable configured Qdrant server raises or
      reports a clear error instead of silently returning an empty in-memory
      client.
- [ ] Database initialization is exercised twice through a mocked or isolated
      connection and the test verifies that the second run succeeds without a
      destructive drop or duplicate-schema failure.
- [ ] Model-download tests use a temporary model directory, prove that a
      complete existing cache causes no Hugging Face download call, and cover
      the incomplete-cache path without making a real network request.
- [ ] Ingestion tests verify configured batch size/collection handling and
      distinguish empty, populated, and force-reindex cases without embedding
      the real dataset.
- [ ] `uv run pytest` passes with network access disabled and without Docker,
      PostgreSQL, Qdrant, an OpenAI key, or model files; the tests do not add a
      runtime dependency.

## Out of scope

- [ ] Building or changing the Docker image and Compose services, covered by
      [TASK-002](002-reproducible-image.md),
      [TASK-004](004-postgres-schema.md),
      [TASK-005](005-qdrant-connectivity.md),
      [TASK-006](006-model-preparation.md),
      [TASK-007](007-qdrant-indexing.md), and
      [TASK-008](008-compose-application.md).
- [ ] Operator runbook content and real clean-stack/restart evidence, moved to
      [TASK-009](009-operator-documentation.md) and
      [TASK-011](011-compose-lifecycle.md).

## Constraints

- Use pytest and the existing development dependencies; do not add a package
  without approval.
- Mock network clients and filesystem/download boundaries rather than calling
  Hugging Face, OpenAI, PostgreSQL, or Qdrant.
- Keep tests isolated from the tracked dataset and from repository model
  caches; use temporary paths where files are required.
- Assert observable calls, configuration, and state transitions rather than
  only asserting that no exception was raised.
