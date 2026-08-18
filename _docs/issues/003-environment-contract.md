# TASK-003 — Define the container environment contract

Status: Groomed

## Goal

An operator can copy a safe example file to configure the Compose stack, and
each service has an unambiguous name, default, and meaning for every required
runtime variable.

## Acceptance criteria

- [ ] A tracked `.env.example` contains a placeholder (not a usable secret)
      for `OPENAI_API_KEY`.
- [ ] `.env.example` documents `STREAMLIT_PORT`,
      `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`,
      `POSTGRES_PASSWORD`, `QDRANT_URL`, `QDRANT_COLLECTION`,
      `QDRANT_BATCH_SIZE`, `QDRANT_FORCE_REINDEX`, and `EMBEDDING_MODEL`.
- [ ] The example gives the Compose defaults `POSTGRES_HOST=postgres` and
      `QDRANT_URL=http://qdrant:6333`, identifies them as internal service
      names, and uses `tw3k_transcripts` and
      `Xenova/all-MiniLM-L6-v2` as the collection and model defaults.
- [ ] The example states the expected port, boolean, and batch-size formats,
      including that `QDRANT_FORCE_REINDEX` is false by default.
- [ ] `.env.example` contains no real key, password, token, or local machine
      path, and the real `.env` file remains ignored by Git.
- [ ] Every variable in the example is assigned to a consuming service or
      script in a short comment or linked documentation, so an operator can
      tell whether a value is host-facing or internal-only.

## Out of scope

- [ ] Reading and validating these values in application code, moved to
      [TASK-004](004-postgres-schema.md),
      [TASK-005](005-qdrant-connectivity.md),
      [TASK-006](006-model-preparation.md),
      [TASK-007](007-qdrant-indexing.md), and
      [TASK-008](008-compose-application.md).
- [ ] First-start, shutdown, reindex, and synthetic-data instructions, moved
      to [TASK-009](009-operator-documentation.md).
- [ ] Automated behavior tests and live-stack validation, moved to
      [TASK-010](010-container-tests.md) and
      [TASK-011](011-compose-lifecycle.md).

## Constraints

- Keep real credentials out of Git and out of the Docker build context.
- Use the variable names in this contract consistently; do not introduce a
  second spelling for the same setting.
- Defaults must work with the Compose service names, while local non-Compose
  overrides remain possible through `.env`.
- Do not add a dependency merely to load or document environment variables.
