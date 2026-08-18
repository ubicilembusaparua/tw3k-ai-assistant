# TASK-009 — Document startup, persistence, and opt-in demo data

Status: Groomed

## Goal

A new operator can configure, start, inspect, stop, reset, reindex, and
optionally seed the Compose stack by following the repository documentation,
without manually running required initialization scripts.

## Acceptance criteria

- [ ] The README or a linked operator document names the prerequisites,
      explains how to copy `.env.example` to `.env` and set the OpenAI key,
      and gives the first-start command from the repository root.
- [ ] The first-start section explains that model download and Qdrant indexing
      can take time and require network access to Hugging Face, and lists the
      expected `model-init`, `db-init`, and `qdrant-init` completion stages.
- [ ] The documentation gives the Streamlit URL using `STREAMLIT_PORT` and
      provides service-specific log commands for diagnosing startup failures.
- [ ] It distinguishes `docker compose down` (named volumes retained) from
      `docker compose down -v` (application data and cached models removed),
      including a clear warning before the destructive command.
- [ ] It gives an explicit force-reindex command, states that it replaces the
      stored vectors, and explains how to return to the normal non-force
      startup behavior.
- [ ] It gives a separate command for `scripts/seed_synthetic_requests.py`,
      states that schema initialization must already have succeeded, and says
      that the command is optional demo data only.
- [ ] The normal `docker compose up` instructions contain no manual model,
      database, Qdrant, or synthetic-data initialization step and do not
      instruct the operator to expose internal service ports.

## Out of scope

- [ ] Implementing the image, services, environment parsing, model/index
      behavior, and dependency graph, covered by
      [TASK-002](002-reproducible-image.md),
      [TASK-003](003-environment-contract.md),
      [TASK-004](004-postgres-schema.md),
      [TASK-005](005-qdrant-connectivity.md),
      [TASK-006](006-model-preparation.md),
      [TASK-007](007-qdrant-indexing.md), and
      [TASK-008](008-compose-application.md).
- [ ] Adding behavior tests or collecting clean-stack evidence, moved to
      [TASK-010](010-container-tests.md) and
      [TASK-011](011-compose-lifecycle.md).

## Constraints

- Keep the documented commands aligned with the checked-in Compose service
  names and environment contract.
- Never put a real secret in documentation or example configuration.
- Make synthetic data explicitly opt in; do not hide it behind startup.
- Explain persistence and destructive reset accurately; do not recommend
  volume removal as routine troubleshooting.
