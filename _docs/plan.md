# Docker Compose Isolation Plan

This plan containerizes the existing Streamlit RAG application and its local
runtime dependencies.

## Agreed Scope

- Run the Streamlit app, PostgreSQL, and Qdrant through Docker Compose.
- Automatically download the ONNX embedding model on first startup.
- Automatically initialize Qdrant when its collection is empty.
- Automatically initialize the PostgreSQL schema with empty tables.
- Track `data/tw3k_dataset.jsonl` in Git and include it in the application image.
- Keep `dataset_builder` ignored and outside the Docker build context.
- Persist PostgreSQL data, Qdrant data, and the ONNX model cache in named volumes.
- Expose Streamlit and bind Qdrant's dashboard to localhost; keep PostgreSQL
  internal to the Compose network.
- Keep synthetic dashboard data out of normal startup.
- Do not add Python dependencies without approval; use the existing `uv.lock`.

## Target Compose Architecture

The services should start in this order:

```text
postgres --healthy --> db-init --completed --\
                                             \
                                              app
model-init --completed --> qdrant --healthy --> qdrant-init --completed --/
```

### Runtime services

- **`app`**: Build the project image and run Streamlit on `0.0.0.0`. It must
  depend on successful completion of `db-init` and `qdrant-init`.
- **`postgres`**: Run a pinned PostgreSQL image with a named data volume and a
  `pg_isready` health check. It is reachable by the application as `postgres`.
- **`qdrant`**: Run a pinned Qdrant image with a named storage volume and an
  HTTP health check. It is reachable by the application as `qdrant`.

### One-shot initialization services

- **`model-init`**: Reuse the project image and run `scripts/download_model.py`.
  Mount the shared model volume at `/app/models`. The script must remain
  idempotent and download `Xenova/all-MiniLM-L6-v2` only when its files are
  missing.
- **`qdrant-init`**: Wait for `model-init` and a healthy Qdrant service, then
  run the ingestion script against `http://qdrant:6333`. The default behavior
  must skip indexing when `tw3k_transcripts` already contains points.
- **`db-init`**: Wait for a healthy PostgreSQL service, then run the existing
  database initialization code. Schema creation must be safe to repeat.

## Implementation Phases

### Phase 1: Container build and repository inputs

- [ ] Keep `/data/tw3k_dataset.jsonl` tracked; retain the
  `/dataset_builder/` ignore rule.
- [ ] Add a `.dockerignore` that excludes secrets, virtual environments,
  caches, generated models, and `dataset_builder`, while retaining the tracked
  dataset.
- [ ] Add a `Dockerfile` using the existing Python/`uv` project configuration.
  Install from `uv.lock` and do not add dependencies.
- [ ] Copy the application source, scripts, and `data/tw3k_dataset.jsonl` into the
  image. Keep model files in the runtime volume rather than baking them into
  the image.
- [ ] Run the application as a non-root user where supported by the chosen
  base image.

### Phase 2: Compose services, networking, and persistence

- [ ] Replace the Qdrant-only Compose definition with the complete service
  graph: `app`, `postgres`, `qdrant`, `model-init`, `qdrant-init`, and
  `db-init`.
- [ ] Add named volumes for PostgreSQL data, Qdrant storage, and ONNX models.
- [ ] Add health checks and `depends_on` conditions so the app cannot start
  before both initialization jobs succeed.
- [ ] Publish the configured Streamlit port and bind Qdrant's dashboard to
  localhost. Keep PostgreSQL internal to the Compose network.
- [ ] Pin database image versions instead of using floating `latest` tags.

### Phase 3: Configuration and initialization behavior

- [ ] Add a safe `.env.example` documenting the required OpenAI, PostgreSQL,
  application-port, Qdrant, and embedding-model settings. Never commit real
  secrets.
- [ ] Make the application read `QDRANT_URL`, `QDRANT_COLLECTION`, and the
  embedding model setting from the environment, using Compose service names
  rather than `localhost`.
- [ ] Make ingestion read the Qdrant collection, batch size, and reindex flag
  from configuration. Default `QDRANT_FORCE_REINDEX` to false.
- [ ] Add an explicit reindex workflow for dataset changes. Normal
  `docker compose up` must not rebuild an existing collection.
- [ ] Change external Qdrant connection failures to fail clearly instead of
  silently falling back to an in-memory client in the containerized path.
- [ ] Make `tw3k_ai_assistant.database.initialization` idempotent, including `CREATE TABLE IF NOT EXISTS` and
  safe repeated feedback-table initialization.
- [ ] Keep `scripts/seed_synthetic_requests.py` as an explicit, documented
  opt-in operation only.

### Phase 4: Application image and operational documentation

- [ ] Configure the app command to bind Streamlit to `0.0.0.0` and the
  configured container port.
- [ ] Document first startup, logs, stopping, persistent-volume behavior, and
  the explicit reindex and synthetic-data commands.
- [ ] Document that the first startup requires network access to Hugging Face
  and may take time while the ONNX model and Qdrant vectors are prepared.
- [ ] Document `docker compose down` versus `docker compose down -v`; the
  latter intentionally removes application data and cached models.

### Phase 5: Validation

- [ ] Run `uv run pytest` before and after the container changes.
- [ ] Validate the file with `docker compose config`.
- [ ] Build and start from a clean checkout with `docker compose up --build`.
- [ ] Verify that the first run downloads the model, creates the database
  schema, indexes Qdrant, and makes Streamlit available.
- [ ] Restart the stack and verify that model download and Qdrant indexing are
  skipped when the persisted data already exists.
- [ ] Verify the explicit reindex path and confirm that the application waits
  for it to complete.
- [ ] Verify that PostgreSQL and Qdrant data survive a normal `down` and are
  removed only with the volume-removal workflow.

## Acceptance Criteria

1. A user with Docker, the tracked repository, and a configured OpenAI key can
   start the complete application with one Compose command.
2. No manual model-download, database-init, or Qdrant-ingestion command is
   required on first startup.
3. Repeated startup is safe and does not re-embed an existing Qdrant
   collection unless reindexing is explicitly requested.
4. The ONNX embedder is loaded from the shared model volume by both ingestion
   and the application.
5. PostgreSQL tables are created automatically and remain empty unless the
   optional synthetic-data command is run.
6. The application uses Compose service names and does not silently use
   in-memory Qdrant data after a connection failure.
7. The existing test suite passes and `dataset_builder` remains ignored.
