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

Qdrant and application commands are documented as their implementation phases
are added.
