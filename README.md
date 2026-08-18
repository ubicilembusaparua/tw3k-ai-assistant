# TW3K AI Assistant

The assistant answers Total War: Three Kingdoms questions with a local ONNX
embedding model, Qdrant retrieval, and a Streamlit interface. The repository
also contains the BM25, dense, and hybrid retrieval components used by the
evaluation scripts.

## Compose prerequisites

Install Docker Desktop or Docker Engine with the Compose v2 plugin, Git, and
an OpenAI API key. The first startup needs network access to Hugging Face to
download the selected ONNX model. Qdrant indexing also runs on the first
startup and can take several minutes.

## Start the stack

Run these commands from the repository root:

```bash
cp .env.example .env
# Edit .env and replace OPENAI_API_KEY with your key.
docker compose up --build -d
```

On PowerShell, the copy command is `Copy-Item .env.example .env`. The normal
Compose startup performs the required initialization jobs automatically:

1. `model-init` downloads or reuses the model in the shared model volume.
2. `db-init` waits for PostgreSQL health and creates the empty schema.
3. `qdrant-init` waits for the model and Qdrant health, then indexes the
   tracked dataset when the collection is empty.
4. `app` starts Streamlit only after `db-init` and `qdrant-init` succeed.

Only Streamlit is published to the host. PostgreSQL and Qdrant remain on the
internal Compose network. Open `http://localhost:${STREAMLIT_PORT}` in a
browser; the default `STREAMLIT_PORT` is `8501`.

## Inspect and diagnose

```bash
docker compose ps
docker compose logs -f model-init
docker compose logs -f db-init
docker compose logs -f qdrant-init
docker compose logs -f app
docker compose logs -f postgres
docker compose logs -f qdrant
```

The initialization jobs must finish successfully. If one fails, inspect its
service-specific log before retrying; the app dependency graph prevents a
failed initialization from looking like a healthy application.

## Stop, persist, and reset

```bash
docker compose down
```

`docker compose down` stops the services but retains the named PostgreSQL,
Qdrant, and model volumes. A later `docker compose up -d` reuses those files
and vectors; normal startup skips indexing for a populated collection.

To intentionally remove all application data and cached model files:

```bash
docker compose down -v
```

WARNING: `down -v` is destructive. It removes the named volumes, including
conversation/feedback data, Qdrant vectors, and the downloaded model cache.
Use it only with data that can be recreated.

## Force a vector rebuild

After changing `tw3k_dataset.jsonl`, keep the stack running and run:

```bash
docker compose run --rm --no-deps -e QDRANT_FORCE_REINDEX=true qdrant-init
```

This explicitly deletes and recreates the configured Qdrant collection, then
indexes the current dataset. The temporary `-e` override does not change the
normal `QDRANT_FORCE_REINDEX=false` setting in `.env`; subsequent ordinary
startup returns to skip behavior for a populated collection.

## Optional synthetic dashboard data

The application schema must already have initialized successfully. Synthetic
requests and feedback are optional demo data and are never created by normal
startup:

```bash
docker compose exec app python scripts/seed_synthetic_requests.py --replace
```

The `--replace` flag intentionally replaces prior synthetic rows. Do not run
this command when the dashboard should remain empty.

## Local development

```bash
uv sync
uv run pytest
uv run streamlit run app.py
```

The local Streamlit command uses the same application modules; configure the
`POSTGRES_*`, `QDRANT_*`, `EMBEDDING_MODEL`, and `OPENAI_API_KEY` values in a
local, ignored `.env` when external services are required.
