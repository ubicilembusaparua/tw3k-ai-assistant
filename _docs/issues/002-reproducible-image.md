# TASK-002 — Build a reproducible application image

Status: Groomed

## Goal

The tracked application inputs can be built into a deterministic Docker image
from the repository root. The image contains the source, scripts, lockfile,
and tracked dataset, but never contains secrets, generated models, or local
development state.

## Acceptance criteria

- [ ] `tw3k_dataset.jsonl` is tracked by Git and is not excluded by
      `.gitignore`; the `/dataset_builder/` rule remains present.
- [ ] A root `.dockerignore` excludes at least `.env`, virtual environments,
      Python caches, test caches, generated `models/`, `.git/`, and
      `dataset_builder/`, while leaving `tw3k_dataset.jsonl` in the build
      context.
- [ ] `Dockerfile` installs the project from `uv.lock` without resolving a
      different dependency set or adding a dependency to `pyproject.toml`.
- [ ] A successful image build has a stable working directory at `/app` and
      contains the application modules, `src/`, `scripts/`, and
      `tw3k_dataset.jsonl` at paths used by the existing commands.
- [ ] The image does not contain `.env` or any generated model files, and its
      default application user is non-root.
- [ ] `docker build .` succeeds from a checkout that has no local model cache
      and does not require a secret to be passed as a build argument.

## Out of scope

- [ ] The environment-variable contract, moved to
      [TASK-003](003-environment-contract.md).
- [ ] PostgreSQL, Qdrant, model-download, Compose dependency, operator
      documentation, test, and lifecycle work, moved to
      [TASK-004](004-postgres-schema.md),
      [TASK-005](005-qdrant-connectivity.md),
      [TASK-006](006-model-preparation.md),
      [TASK-007](007-qdrant-indexing.md),
      [TASK-008](008-compose-application.md),
      [TASK-009](009-operator-documentation.md),
      [TASK-010](010-container-tests.md), and
      [TASK-011](011-compose-lifecycle.md).

## Constraints

- Use the existing `pyproject.toml` and `uv.lock`; dependency additions
  require approval.
- Use the repository's Python requirement (`>=3.11`) and keep the build
  context at the repository root.
- Do not bake the ONNX model or the dataset-builder output into the image.
- Preserve the tracked dataset exactly; only its ignore/build-context status
  is in scope here.
- Keep secrets out of both image layers and build logs.
