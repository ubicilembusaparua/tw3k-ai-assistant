# Docker Compose Implementation Backlog

## 1. Set up an empty project with a passing test
Goal: Establish a minimal Python project that runs one passing test with `uv run pytest`.
Description: Create the smallest project structure needed for the test runner, including the project metadata and one deterministic test. Verify that the test passes from a clean environment without adding application behavior.

## 2. Build a reproducible application image
Goal: Make the tracked dataset and application dependencies available in a deterministic Docker image.
Description: Remove `tw3k_dataset.jsonl` from `.gitignore`, retain the `dataset_builder` ignore rule, and add `.dockerignore` rules for secrets, environments, caches, and generated models. Add a `Dockerfile` that installs from `uv.lock`, copies the application source, scripts, and dataset, and defines a stable working directory.

## 3. Define the container environment contract
Goal: Document the environment variables required by the application and Compose services.
Description: Add a safe `.env.example` covering the OpenAI key, Streamlit port, PostgreSQL settings, Qdrant settings, and ONNX model name. Keep real secrets out of Git and document which values are internal Compose service names.

## 4. Add PostgreSQL and automatic schema initialization
Goal: Start a persistent PostgreSQL service and create the application schema automatically.
Description: Add a pinned PostgreSQL service with environment-based credentials, a named data volume, and a `pg_isready` health check, then make `db_init.py` safe to run repeatedly without deleting existing data. Add the one-shot `db-init` service so it waits for PostgreSQL readiness, creates empty conversations and feedback tables, and reports initialization failures clearly.

## 5. Add Qdrant and explicit application connectivity
Goal: Run Qdrant as a durable internal service and make connection failures visible.
Description: Add a pinned Qdrant service with a named storage volume and an image-compatible health check, keeping its ports internal by default. Configure the retrieval pipeline to use `QDRANT_URL` and `QDRANT_COLLECTION` with `http://qdrant:6333` inside Compose, and prevent unavailable external Qdrant connections from silently falling back to in-memory data.

## 6. Add automatic ONNX model preparation
Goal: Make the selected ONNX embedder available automatically to ingestion and the application.
Description: Make `scripts/download.py` and the embedder configuration accept a model setting while defaulting to `Xenova/all-MiniLM-L6-v2`, and verify that downloads are idempotent under `models/<model-name>`. Add a one-shot `model-init` service that mounts a named volume at `/app/models` and completes before any service that needs the model starts.

## 7. Add idempotent Qdrant initialization and reindexing
Goal: Populate an empty Qdrant collection automatically while providing an explicit rebuild workflow.
Description: Update `ingest_qdrant.py` to read the Qdrant URL, collection, batch size, and force-reindex setting, skipping an existing collection by default. Add `qdrant-init` so ingestion waits for `model-init` and a healthy Qdrant service, then document a force-enabled command for rebuilding vectors after dataset changes.

## 8. Start the application with the complete Compose dependency graph
Goal: Start Streamlit only after PostgreSQL and Qdrant initialization have succeeded.
Description: Add the `app` service, bind Streamlit to `0.0.0.0`, and publish only the configured application port. Wire `depends_on` completion conditions so the app waits for `db-init` and `qdrant-init`, and declare the PostgreSQL, Qdrant, and ONNX named volumes for the services that use them.

## 9. Document startup, persistence, and opt-in demo data
Goal: Let a new operator run and manage the stack without manually running initialization scripts.
Description: Document first-run model download and vector indexing, service logs, the Streamlit URL, normal shutdown, volume reset, and the explicit reindex command. Keep `scripts/seed_synthetic_requests.py` outside normal startup and document it as an optional command for intentionally creating demo dashboard data.

## 10. Add tests for container configuration behavior
Goal: Protect environment parsing and repeatable initialization with deterministic tests.
Description: Add unit tests for Qdrant environment configuration, force-reindex defaults, database initialization repeatability, and model-download skipping when files exist. Keep the tests independent of Docker, external APIs, and real OpenAI credentials.

## 11. Validate the complete Compose lifecycle
Goal: Prove that a clean checkout can start and reuse the full application stack.
Description: Run `docker compose config`, build the image, and start the stack from a clean state. Verify first-run initialization, repeated startup without duplicate model download or indexing, explicit reindexing, service availability, volume persistence, and the full `uv run pytest` suite.
