# TASK-004 — Add PostgreSQL and automatic schema initialization

Status: Groomed

## Goal

Compose starts a persistent PostgreSQL service and creates the existing
conversation and feedback schema automatically. Initialization is safe to
repeat, never seeds fake dashboard data, and fails visibly when the database
cannot be initialized.

## Acceptance criteria

- [ ] The `postgres` service uses a pinned PostgreSQL image tag, obtains its
      database name and credentials from the environment contract, and mounts
      a named data volume.
- [ ] PostgreSQL has a `pg_isready` health check that uses the configured
      database and user; PostgreSQL does not publish a host port by default.
- [ ] A fresh database run creates both `conversations` and `feedback` with
      the columns and foreign-key relationship required by the existing save
      and query modules.
- [ ] Running the initializer twice against the same database exits 0 on the
      second run, produces no duplicate-table or migration error, and leaves
      pre-existing conversation and feedback rows intact.
- [ ] The default initialization path never drops tables and leaves both
      tables empty on a fresh deployment; any explicit destructive option is
      not used by the Compose `db-init` service.
- [ ] The one-shot `db-init` service waits for PostgreSQL health, exits 0 only
      after schema creation succeeds, and exits non-zero with a useful error
      when PostgreSQL is unavailable or schema creation fails.
- [ ] `scripts/seed_synthetic_requests.py` is not invoked by the database
      image, initializer, or normal Compose dependency graph.

## Out of scope

- [ ] Qdrant connectivity and its health/storage contract, moved to
      [TASK-005](005-qdrant-connectivity.md).
- [ ] ONNX model preparation, vector ingestion, the application service, and
      the complete dependency graph, moved to
      [TASK-006](006-model-preparation.md),
      [TASK-007](007-qdrant-indexing.md), and
      [TASK-008](008-compose-application.md).
- [ ] Operator instructions, deterministic unit tests, and end-to-end
      lifecycle evidence, moved to
      [TASK-009](009-operator-documentation.md),
      [TASK-010](010-container-tests.md), and
      [TASK-011](011-compose-lifecycle.md).

## Constraints

- Preserve the current schema and legacy feedback migration behavior unless
  it is required to make a repeat run safe.
- Use the existing `psycopg` connection code and SQL; do not add an ORM or a
  migration package without approval.
- Keep credentials in environment variables and keep PostgreSQL internal to
  the Compose network.
- Idempotence must be achieved without deleting user data on the normal path.
