# TASK-001 — Establish a runnable project baseline

Status: Groomed

## Goal

A clean checkout has the minimum project metadata and one deterministic test
needed to establish a working Python test command. The baseline test can run
without the application, Docker, databases, model downloads, API keys, or
network services.

## Acceptance criteria

- [ ] From the repository root, `uv sync` completes using the checked-in
      project metadata and lockfile.
- [ ] `uv run pytest tests/test_home.py` exits with status 0 and reports at
      least one passing test.
- [ ] `uv run pytest` also exits with status 0 from a clean environment.
- [ ] The baseline test is deterministic and does not require Docker,
      PostgreSQL, Qdrant, OpenAI credentials, Hugging Face access, or a
      downloaded model.
- [ ] The project metadata and test are sufficient for the passing baseline;
      no application behavior is added solely to make this issue pass.

## Out of scope

- [ ] Building the Docker image and controlling its build context, moved to
      [TASK-002](002-reproducible-image.md).
- [ ] Defining runtime variables, services, initialization jobs, application
      startup, operator documentation, container-specific tests, and full
      lifecycle validation, moved to [TASK-003](003-environment-contract.md),
      [TASK-004](004-postgres-schema.md),
      [TASK-005](005-qdrant-connectivity.md),
      [TASK-006](006-model-preparation.md),
      [TASK-007](007-qdrant-indexing.md),
      [TASK-008](008-compose-application.md),
      [TASK-009](009-operator-documentation.md),
      [TASK-010](010-container-tests.md), and
      [TASK-011](011-compose-lifecycle.md).

## Constraints

- Use `uv` and the existing Python version declaration; do not add a package
  without approval.
- Keep the test independent of external state and safe to run repeatedly.
- Keep `dataset_builder` ignored as required by the repository instructions.
- Do not turn this baseline issue into a Docker, database, or application
  feature implementation.
