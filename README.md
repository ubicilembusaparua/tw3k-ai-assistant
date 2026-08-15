# TW3K AI Assistant

A local retrieval-augmented generation assistant for asking questions about the
Total War: Three Kingdoms guidance contained in the project's transcript JSONL
dataset.

## Development setup

1. Install Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).
2. Copy `.env.example` to `.env` and fill in values required by the command you
   intend to run. Do not commit `.env`.
3. Install the locked dependencies:

   ```powershell
   uv sync
   ```

4. Run the test suite:

   ```powershell
   uv run pytest
   ```

## Local Qdrant

Start Qdrant in the background with one command:

```powershell
docker compose up -d qdrant
```

The service listens at `http://localhost:6333`; its dashboard is available at
`http://localhost:6333/dashboard`. The Compose health check probes the local
REST port, and the `qdrant_data` named volume keeps indexed vectors across
container restarts and ordinary `docker compose down` operations.

Inspect status and logs:

```powershell
docker compose ps qdrant
docker compose logs qdrant
```

Stop the service without deleting indexed data:

```powershell
docker compose down
```

Use `.env.example` as the complete list of application settings. Defaults allow
validation and local retrieval to start without credentials. Answer generation
requires `OPENAI_API_KEY`; a missing key is reported separately from an
unreachable Qdrant service.

## Validate and index transcripts

Validate a JSONL file without loading models or contacting Qdrant:

```powershell
uv run tw3k-index tw3k_dataset.jsonl --validate-only
```

With Qdrant running, build or update the local index:

```powershell
uv run tw3k-index tw3k_dataset.jsonl
```

Records are preflighted before any write. Embeddings and Qdrant upserts are
streamed in bounded batches, and stable point IDs make reruns update existing
chunks instead of duplicating them. Each command prints validated, embedded,
upserted, skipped, and failed counts as JSON and exits nonzero if indexing is
incomplete.

## Inspect retrieval

Retrieve and rerank evidence without making an OpenAI request:

```powershell
uv run tw3k-retrieve "How should I manage food in the early game?"
```

The diagnostic JSON includes dense-retrieval and reranking scores, selected
transcript text, and exact timestamp links. Empty or irrelevant results return a
deterministic `insufficient_evidence` status and a nonzero exit code.

Answer generation uses the configured `OPENAI_MODEL` only after retrieval finds
sufficient evidence. The application supplies its own `SOURCE_N` identifiers,
requires every substantive claim to cite them, rejects unknown citation IDs, and
maps accepted citations to the corresponding exact YouTube timestamp. OpenAI
and Qdrant failures are surfaced as service errors rather than answers.

## Run the local web app

Set `OPENAI_API_KEY` in the process environment, ensure Qdrant is running and
indexed, then launch:

```powershell
uv run uvicorn src.web:app --reload
```

Open `http://127.0.0.1:8000`. The readiness endpoint is
`http://127.0.0.1:8000/api/readiness`. It reports the application, Qdrant and
index count, cached local models, and OpenAI configuration independently.
Model loading and all Qdrant/OpenAI network calls occur in the service layer;
the application does not hold a database transaction around that work.
