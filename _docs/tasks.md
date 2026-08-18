# Docker Compose implementation issues

The grouped backlog has been groomed into the implementation-ready issues
below. Each issue follows `_docs/task-template.md`; links in the issues make
scope transfers explicit.

| Issue | Title | Depends on |
| --- | --- | --- |
| [TASK-001](issues/001-project-baseline.md) | Establish a runnable project baseline | — |
| [TASK-002](issues/002-reproducible-image.md) | Build a reproducible application image | TASK-001 |
| [TASK-003](issues/003-environment-contract.md) | Define the container environment contract | TASK-002 |
| [TASK-004](issues/004-postgres-schema.md) | Add PostgreSQL and automatic schema initialization | TASK-002, TASK-003 |
| [TASK-005](issues/005-qdrant-connectivity.md) | Add Qdrant and explicit application connectivity | TASK-002, TASK-003 |
| [TASK-006](issues/006-model-preparation.md) | Add automatic ONNX model preparation | TASK-002, TASK-003 |
| [TASK-007](issues/007-qdrant-indexing.md) | Add idempotent Qdrant initialization and reindexing | TASK-005, TASK-006 |
| [TASK-008](issues/008-compose-application.md) | Start the application with the complete Compose graph | TASK-004, TASK-005, TASK-006, TASK-007 |
| [TASK-009](issues/009-operator-documentation.md) | Document startup, persistence, and opt-in demo data | TASK-008 |
| [TASK-010](issues/010-container-tests.md) | Add tests for container configuration behavior | TASK-004, TASK-005, TASK-006, TASK-007 |
| [TASK-011](issues/011-compose-lifecycle.md) | Validate the complete Compose lifecycle | TASK-008, TASK-009, TASK-010 |

Implementation should follow the dependency order above. TASK-011 is the
validation gate for the complete stack, not a substitute for the acceptance
criteria in the earlier issues.
