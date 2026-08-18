# TASK-008 — Start the application with the complete Compose graph

Status: Groomed

## Goal

One Compose project starts Streamlit only after the database schema, model, and
Qdrant index initialization jobs have succeeded. The host exposes only the
configured Streamlit port while all persistent dependencies remain internal
and durable.

## Acceptance criteria

- [ ] `compose.yaml` defines the six services `app`, `postgres`, `qdrant`,
      `model-init`, `qdrant-init`, and `db-init`.
- [ ] The `app` command runs Streamlit for `app.py` on `0.0.0.0` and uses the
      configured `STREAMLIT_PORT` inside the container.
- [ ] Only the configured Streamlit port is published to the host; PostgreSQL
      and Qdrant have no default host port mappings.
- [ ] `app` has completion-based dependencies on successful `db-init` and
      `qdrant-init`, and cannot start when either job exits non-zero.
- [ ] The graph also enforces that `qdrant-init` waits for successful
      `model-init` and healthy Qdrant, while `db-init` waits for healthy
      PostgreSQL.
- [ ] Named volumes are declared and mounted for PostgreSQL data, Qdrant
      storage, and ONNX models; the application and ingestion paths read the
      model from the shared model volume.
- [ ] The app receives the OpenAI key and internal `POSTGRES_HOST`/
      `QDRANT_URL` settings through runtime environment configuration, and no
      secret is copied into the image or committed Compose file.
- [ ] A dependency failure prevents an apparently healthy app from starting;
      after all completion conditions succeed, the app process remains
      running for Compose to supervise.

## Out of scope

- [ ] The individual PostgreSQL, Qdrant, model, and ingestion behaviors,
      specified in [TASK-004](004-postgres-schema.md),
      [TASK-005](005-qdrant-connectivity.md),
      [TASK-006](006-model-preparation.md), and
      [TASK-007](007-qdrant-indexing.md).
- [ ] Startup/runbook content, isolated behavior tests, and full clean-stack
      evidence, moved to [TASK-009](009-operator-documentation.md),
      [TASK-010](010-container-tests.md), and
      [TASK-011](011-compose-lifecycle.md).

## Constraints

- Use Compose dependency completion conditions and named volumes; do not
  replace readiness with arbitrary sleeps.
- Keep service-to-service addresses on the Compose network and do not expose
  database ports by default.
- Reuse the image built in TASK-002 and the environment names in TASK-003.
- Keep synthetic dashboard seeding outside the dependency graph.
- Do not add Python dependencies without approval.
