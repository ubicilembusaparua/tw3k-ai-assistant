# Docker Compose Implementation Backlog

## 1. Set up an empty project with a passing test
Goal: Establish a minimal Python project that runs one passing test with `uv run pytest`.
Description: Create the smallest project structure needed for the test runner, including the project metadata and one deterministic test. Verify that the test passes from a clean environment without adding application behavior.

## 2. Track the dataset and define the Docker build context
Goal: Make `tw3k_dataset.jsonl` available to Docker while keeping generated data excluded.
Description: Remove the dataset from `.gitignore` and confirm it is tracked by Git. Add `.dockerignore` rules that exclude secrets, virtual environments, caches, downloaded models, and `dataset_builder` while retaining the dataset.

## 3. Create a reproducible application image
Goal: Build the Python application from the existing project configuration and lockfile.
Description: Add a `Dockerfile` that uses the repository's Python version and installs dependencies from `uv.lock` without adding packages. Copy the application source, scripts, and tracked dataset into the image, and define a stable working directory.

## 4. Define the container environment contract
Goal: Document the environment variables required by the application and Compose services.
Description: Add a safe `.env.example` covering the OpenAI key, Streamlit port, PostgreSQL settings, Qdrant settings, and ONNX model name. Keep real secrets out of Git and document which values are internal Compose service names.

## 5. Add PostgreSQL with health checks and persistent storage
Goal: Run PostgreSQL as a Compose service whose data survives normal restarts.
Description: Add a pinned PostgreSQL image, credentials from environment variables, a named data volume, and a `pg_isready` health check. Keep PostgreSQL reachable only on the internal Compose network by default.

## 6. Make database initialization repeatable
Goal: Allow the database schema to be created safely on every startup.
Description: Update `db_init.py` so conversations and feedback tables can be initialized more than once without failing or deleting existing data. Preserve the existing schema and migration behavior, and do not seed synthetic dashboard rows.

## 7. Add the PostgreSQL initialization service
Goal: Initialize the PostgreSQL schema automatically after the database is ready.
Description: Add a one-shot `db-init` Compose service that reuses the application image and runs the idempotent database initializer. Configure it to wait for PostgreSQL health and report failure instead of allowing the app to start with a missing schema.

## 8. Add Qdrant with health checks and persistent storage
Goal: Run Qdrant as a durable internal Compose service.
Description: Replace the current Qdrant-only definition with a pinned Qdrant image, a named storage volume, and an image-compatible HTTP health check. Keep Qdrant's ports internal unless a later operational requirement explicitly exposes them.

## 9. Make Qdrant connection settings explicit
Goal: Ensure the application uses the Compose Qdrant service and fails clearly when it is unavailable.
Description: Make the retrieval pipeline read `QDRANT_URL` and `QDRANT_COLLECTION` from the environment, using `http://qdrant:6333` inside Compose instead of `localhost`. Prevent the containerized path from silently switching to an in-memory client after an external Qdrant connection fails.

## 10. Make ONNX model downloading configurable and verifiable
Goal: Ensure the selected ONNX embedder files are downloaded to the expected path.
Description: Update `scripts/download.py` and the embedder configuration as needed so the model name can be supplied by configuration while defaulting to `Xenova/all-MiniLM-L6-v2`. Verify that tokenizer and ONNX files are downloaded idempotently under `models/<model-name>`.

## 11. Add the model initialization service
Goal: Download the ONNX model automatically before vector ingestion or application startup.
Description: Add a one-shot `model-init` Compose service that runs the download script and mounts a named volume at `/app/models`. Make later startups reuse the existing files instead of downloading the model again.

## 12. Make Qdrant ingestion idempotent and configurable
Goal: Index the tracked dataset only when the target collection is empty unless reindexing is requested.
Description: Update `ingest_qdrant.py` to read the Qdrant URL, collection, batch size, and force-reindex setting from configuration. Keep normal startup non-destructive and provide an explicit force option for rebuilding the collection after dataset changes.

## 13. Add the Qdrant initialization service
Goal: Populate an empty Qdrant collection automatically during Compose startup.
Description: Add a one-shot `qdrant-init` service that waits for `model-init` to complete and Qdrant to pass its health check. Run the idempotent ingestion command against `http://qdrant:6333` and fail the startup sequence if ingestion cannot complete.

## 14. Add the Streamlit application service
Goal: Run the existing Streamlit interface from the project image.
Description: Add an `app` Compose service that starts `app.py`, binds Streamlit to `0.0.0.0`, and publishes only the configured application port. Pass the OpenAI and internal database settings through the environment without copying `.env` or secrets into the image.

## 15. Wire service dependencies and shared volumes
Goal: Guarantee that the application starts only after both database initialization paths succeed.
Description: Configure Compose health checks and `depends_on` completion conditions so `app` waits for `db-init` and `qdrant-init`, while `qdrant-init` waits for Qdrant and `model-init`. Declare named volumes for PostgreSQL data, Qdrant storage, and ONNX models and mount them in the services that need them.

## 16. Add an explicit Qdrant reindex workflow
Goal: Give operators a documented way to rebuild vectors after the dataset changes.
Description: Add a Compose command or documented override that runs `qdrant-init` with force reindex enabled. Ensure the normal `docker compose up` path skips an existing collection and the explicit workflow clearly warns that it replaces stored vectors.

## 17. Keep synthetic dashboard data opt-in
Goal: Prevent fake monitoring records from appearing in a fresh deployment.
Description: Leave `scripts/seed_synthetic_requests.py` outside the normal Compose dependency graph. Document a separate command that runs it only after PostgreSQL schema initialization when demo dashboard data is intentionally needed.

## 18. Document first startup and data lifecycle
Goal: Enable a new user to run, stop, reset, and troubleshoot the Compose stack without manual initialization scripts.
Description: Document the first-run model download and Qdrant indexing delay, service logs, the Streamlit URL, and the explicit reindex and synthetic-data commands. Explain that `docker compose down` preserves named volumes while `docker compose down -v` removes application data and cached models.

## 19. Add tests for container configuration behavior
Goal: Protect the environment parsing and idempotent initialization behavior with fast tests.
Description: Add unit tests for Qdrant environment configuration, force-reindex defaults, database initialization repeatability, and model-download skipping when files exist. Keep the tests deterministic and independent of Docker, external APIs, and real OpenAI credentials.

## 20. Validate the complete Compose lifecycle
Goal: Prove that a clean checkout can start and reuse the full application stack.
Description: Run `docker compose config`, build the image, and start the stack from a clean state. Verify first-run initialization, repeated startup without duplicate model download or indexing, explicit reindexing, service availability, volume persistence, and the full `uv run pytest` suite.
