# TASK-006 — Add automatic ONNX model preparation

Status: Groomed

## Goal

The configured ONNX embedding model is available in a shared named volume
before ingestion or the application starts. The model download is
repeatable, configurable, and safe to rerun without downloading complete
files again.

## Acceptance criteria

- [ ] `scripts/download.py` and the embedder accept `EMBEDDING_MODEL`, with
      `Xenova/all-MiniLM-L6-v2` as the default when it is unset.
- [ ] A complete model is stored below `models/<model-name>` (and at
      `/app/models/<model-name>` in the image) with the tokenizer and ONNX
      files at the paths consumed by `Embedder`; any required ONNX sidecar is
      kept beside its model file.
- [ ] If the required files already exist, a repeat download exits 0 without
      making a Hugging Face download request and reports that the files were
      reused.
- [ ] If only part of the model is present, the command obtains the missing
      files, leaves a usable complete model on success, and exits non-zero with
      a useful error on a download failure rather than reporting false success.
- [ ] The one-shot `model-init` service uses the application image, mounts the
      named model volume at `/app/models`, passes the selected model setting,
      and exits successfully only after the model is ready.
- [ ] The image does not contain a downloaded model; the shared volume is the
      source used by both model consumers.

## Out of scope

- [ ] Qdrant collection creation, indexing, skip behavior, and force
      reindexing, moved to [TASK-007](007-qdrant-indexing.md).
- [ ] Wiring every runtime service and the final `depends_on` graph, moved to
      [TASK-008](008-compose-application.md).
- [ ] Operator instructions, isolated tests, and live lifecycle validation,
      moved to [TASK-009](009-operator-documentation.md),
      [TASK-010](010-container-tests.md), and
      [TASK-011](011-compose-lifecycle.md).

## Constraints

- Use the existing Hugging Face and ONNX implementation; do not add a model
  download library or commit model artifacts.
- Keep the default model name exactly `Xenova/all-MiniLM-L6-v2`.
- Preserve the `models/<model-name>` layout so the local and container paths
  have the same relative structure.
- Do not make a normal startup delete or overwrite a complete cached model.
- The model-init job may require network access on first run, but it must be
  deterministic and idempotent afterward.
