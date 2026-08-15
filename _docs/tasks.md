# Full RAG Assistant Implementation Tasks

## Goal

Extend the existing YouTube dataset builder into a single-user RAG assistant while preserving a strict boundary between dataset creation and retrieval:

```text
YouTube -> SQLite -> JSONL export -> one-time indexer -> Qdrant -> RAG web chat
```

SQLite remains the canonical dataset store. The scraper and exporter must not depend on Qdrant or OpenAI. The RAG system consumes an exported JSONL file and searches across all indexed videos.

## Confirmed product decisions

- Use the OpenAI cloud API for embeddings and answer generation.
- Optimize default model choices for a balance of cost and answer quality, while keeping model names configurable.
- Run Qdrant locally with Docker and persistent storage.
- Support YouTube videos and playlists only in the first version.
- Keep SQLite and JSONL as the dataset-builder outputs.
- Populate Qdrant through a standalone indexing script, not during YouTube ingestion.
- Make indexing idempotent: update existing chunk IDs and add new chunks.
- Search all indexed videos for every question in the first version.
- Make the web dashboard the primary assistant interface.
- Return inline numbered citations linked to exact YouTube timestamps.
- Support one local user with no authentication.
- Persist conversations and allow them to be reopened.
- Start with reliable non-streaming responses; streaming is a later enhancement.
- In the first version, upserting does not delete Qdrant points missing from a newer JSONL file. Add explicit collection rebuild/pruning only as a later feature.

## Phase 0 - Baseline and dependency approval

- [ ] Run the existing test suite and record the baseline behavior.
- [ ] Resolve the duplicated dependency declarations in `pyproject.toml` and `requirements.txt`, documenting `pyproject.toml` plus `uv.lock` as authoritative.
- [ ] Ask for approval before adding the required dependencies, per `AGENTS.md`:
  - OpenAI Python SDK
  - Qdrant Python client
- [ ] After approval, add dependencies only in `pyproject.toml`, run `uv sync`, and commit the updated lockfile.
- [ ] Add a safe example environment file with variable names only; never commit secrets.

Acceptance criteria:

- Existing tests pass before feature work begins.
- The project installs reproducibly with `uv sync`.
- No API keys, cookies, or Qdrant data files are tracked by Git.

## Phase 1 - Configuration and local infrastructure

- [ ] Add centralized typed configuration for:
  - `OPENAI_API_KEY`
  - chat model name
  - embedding model name
  - Qdrant URL
  - Qdrant collection name
  - retrieval result count (`top_k`)
  - optional score threshold
  - maximum context size
- [ ] Fail startup/indexing with clear messages when required settings are missing.
- [ ] Choose `text-embedding-3-small` as the initial cost-conscious embedding default, but keep it configurable.
- [ ] Choose a balanced current OpenAI text model as the initial chat default and isolate it in configuration so it can be changed without code edits.
- [ ] Use the OpenAI Responses API for answer generation.
- [ ] Add Docker Compose configuration for Qdrant.
- [ ] Mount Qdrant data to a named volume so vectors survive container restarts.
- [ ] Add a Qdrant health check and document start, stop, status, and data-reset commands.
- [ ] Add generated environment files and local Qdrant artifacts to `.gitignore` where needed.

Acceptance criteria:

- Qdrant starts with one documented Docker command.
- Restarting the container preserves indexed points.
- Configuration errors identify the missing variable and suggested fix.

## Phase 2 - Define and validate the indexing contract

- [ ] Treat the existing JSONL export as the formal boundary between the dataset builder and RAG system.
- [ ] Define a versioned index-record schema containing at least:
  - `chunk_id`
  - `video_id`
  - `chunk_index`
  - transcript text
  - video title and channel
  - start and end times
  - formatted time
  - timestamp link
  - video URL
- [ ] Add strict validation for every JSONL record before sending data to OpenAI or Qdrant.
- [ ] Report malformed input with the file line number and validation reason.
- [ ] Decide how blank text, duplicate chunk IDs within one file, and missing optional metadata are handled.
- [ ] Preserve stable `chunk_id` values as Qdrant point identities, using a deterministic conversion if Qdrant requires a different ID type.
- [ ] Store the original `chunk_id` and all citation metadata in the Qdrant payload.
- [ ] Record the embedding model and vector dimensions in collection/index metadata so incompatible models cannot silently mix.

Acceptance criteria:

- The exported JSONL can be validated without contacting OpenAI or Qdrant.
- Invalid records stop the index run before partial writes unless an explicit skip-invalid mode is selected.
- A chunk can be traced from SQLite export to its Qdrant point and YouTube timestamp.

## Phase 3 - Build the standalone Qdrant indexing script

- [ ] Add a standalone script with a documented command such as:

  ```text
  uv run python scripts/index_qdrant.py tw3k_dataset.jsonl
  ```

- [ ] Verify Qdrant connectivity before creating embeddings.
- [ ] Create the collection when absent with cosine similarity and dimensions matching the configured embedding model.
- [ ] Refuse to write to an existing collection whose vector size or embedding-model metadata is incompatible.
- [ ] Read JSONL incrementally instead of loading the whole dataset into memory.
- [ ] Batch OpenAI embedding requests and Qdrant upserts.
- [ ] Retry transient OpenAI/Qdrant failures with bounded exponential backoff.
- [ ] Upsert by stable chunk identity so rerunning the script updates existing chunks and adds new chunks without duplication.
- [ ] Do not delete stale points during normal upsert runs in version one.
- [ ] Show progress: records validated, embedded, upserted, skipped, and failed.
- [ ] Exit nonzero on an incomplete run and print a useful summary without exposing secrets.
- [ ] Add a dry-run/validation-only option.
- [ ] Add an optional resume/checkpoint mechanism if the full dataset is large enough to justify it after measurement.

Acceptance criteria:

- Indexing the same JSONL twice leaves the same point count.
- Changing a chunk and rerunning updates its vector and payload.
- Adding a chunk and rerunning increases the point count by one.
- Failed batches can be retried safely without duplicating records.

## Phase 4 - Implement retrieval

- [ ] Add an OpenAI embedding adapter shared by indexing and query embedding.
- [ ] Add a Qdrant repository/service that hides client-specific operations from web routes.
- [ ] Embed each user question with the same configured model used for indexing.
- [ ] Search the entire Qdrant collection using cosine similarity.
- [ ] Make `top_k` and the minimum relevance score configurable.
- [ ] Return ordered retrieval results with similarity scores and complete citation metadata.
- [ ] Deduplicate near-identical overlapping transcript chunks before prompt construction.
- [ ] Enforce a context budget and select the most useful chunks without cutting citation metadata.
- [ ] Return a clear no-evidence result when retrieval quality is below the threshold.
- [ ] Add a diagnostic retrieval endpoint or developer command that returns sources without calling the chat model.

Acceptance criteria:

- A known phrase retrieves the expected video chunk.
- Every retrieved chunk has a valid clickable timestamp link.
- Empty collections, unavailable Qdrant, and low-score searches produce distinct actionable errors.

## Phase 5 - Generate grounded answers with citations

- [ ] Build a RAG orchestration service with explicit stages: retrieve, select context, generate, validate citations, and persist.
- [ ] Create a system prompt requiring the assistant to:
  - answer only from supplied transcript evidence
  - acknowledge when evidence is insufficient
  - avoid inventing video facts or citations
  - use numbered inline citations such as `[1]`
- [ ] Assign citation numbers in application code and provide the model with the allowed source map.
- [ ] Call the OpenAI Responses API with the question, bounded conversation context, and retrieved evidence.
- [ ] Use a structured answer contract that separates answer text from cited source identifiers where practical.
- [ ] Validate that every citation refers to a supplied source.
- [ ] Render each inline citation as a link to the source's `timestamp_link`.
- [ ] Reject or repair invalid citation references before returning the answer.
- [ ] Capture token usage, latency, selected model, retrieval count, and errors for local diagnostics.
- [ ] Keep non-streaming generation as the initial implementation.

Acceptance criteria:

- Answers contain only valid numbered citations.
- Clicking a citation opens the correct YouTube video at the relevant timestamp.
- Questions unsupported by the dataset receive an explicit insufficient-evidence response.
- API failures never result in a fabricated answer.

## Phase 6 - Persist conversations in SQLite

- [ ] Extend SQLite with migration-safe tables for conversations and messages.
- [ ] Store conversation ID, title, creation/update timestamps, role, message content, and ordering.
- [ ] Store assistant citation metadata as structured JSON or normalized rows.
- [ ] Store useful request diagnostics without storing the OpenAI API key or hidden model reasoning.
- [ ] Add repository functions to create, list, load, rename, and delete conversations.
- [ ] Add a deterministic strategy for conversation titles, with optional model-generated titles deferred until needed.
- [ ] Bound history sent to OpenAI by message count or token budget while preserving the complete local transcript.
- [ ] Define deletion confirmation and cascade behavior for messages.
- [ ] Add schema initialization/migration tests against both empty and existing dataset databases.

Acceptance criteria:

- Conversations survive application restarts.
- Reopening a conversation restores messages and citation links in the correct order.
- Existing `videos`, `chunks`, and FTS5 data remain compatible.

## Phase 7 - Add RAG API endpoints

- [ ] Add request/response models and endpoints for:
  - asking a question in a conversation
  - creating a conversation
  - listing conversations
  - loading one conversation and its messages
  - renaming a conversation
  - deleting a conversation
  - checking OpenAI and Qdrant readiness
- [ ] Keep network and model work outside SQLite transactions.
- [ ] Map configuration, Qdrant, OpenAI, validation, and insufficient-evidence failures to appropriate HTTP responses.
- [ ] Add timeouts and bounded retries for external calls.
- [ ] Prevent duplicate submissions from creating duplicate assistant messages.
- [ ] Keep existing dataset-builder endpoints operational unless deliberately deprecated and documented.

Acceptance criteria:

- Endpoint contracts are covered by tests with OpenAI and Qdrant mocked.
- A browser refresh does not duplicate the last request.
- Health/readiness output clearly distinguishes the web app, SQLite, Qdrant, and OpenAI configuration states.

## Phase 8 - Build the web chat dashboard

- [ ] Evolve the existing vanilla HTML/CSS/JavaScript dashboard into the primary chat experience without introducing a frontend framework unless separately approved.
- [ ] Add a conversation sidebar with new, reopen, rename, and delete actions.
- [ ] Add a chat transcript area with distinct user and assistant messages.
- [ ] Add a question composer with submit state and duplicate-submit protection.
- [ ] Render inline citation markers as timestamp links with accessible labels.
- [ ] Provide a source preview showing video title, channel, time range, and relevant transcript text when a citation is selected.
- [ ] Show clear states for loading, no indexed data, Qdrant unavailable, OpenAI errors, and insufficient evidence.
- [ ] Preserve current dataset statistics and ingestion/export controls in a secondary section if they remain useful.
- [ ] Sanitize all transcript and model-provided content before inserting it into the DOM.
- [ ] Make the layout usable on desktop and mobile.
- [ ] Add keyboard-friendly and accessible focus behavior.

Acceptance criteria:

- A local user can start a chat, ask a question, follow a timestamp citation, refresh, and reopen the conversation.
- Untrusted transcript/model text cannot inject HTML or JavaScript.
- The page remains usable when a request takes several seconds.

## Phase 9 - Testing and RAG evaluation

- [ ] Add unit tests for configuration, JSONL validation, stable point IDs, batching, retries, context selection, citation mapping, and conversation storage.
- [ ] Add integration tests against a temporary SQLite database.
- [ ] Mock OpenAI and Qdrant in the default test suite so tests are deterministic and do not incur API costs.
- [ ] Add an opt-in Docker integration test for real Qdrant collection creation, upsert, update, and search.
- [ ] Create a small, committed synthetic JSONL fixture with no copyrighted transcript corpus or secrets.
- [ ] Build a representative evaluation set of questions with expected video/chunk evidence.
- [ ] Measure retrieval hit rate, citation validity, groundedness, latency, and approximate token cost.
- [ ] Tune `top_k`, score threshold, overlap deduplication, context budget, and prompts using the evaluation set.
- [ ] Add regression tests for unsupported questions and prompt-injection text inside transcripts.
- [ ] Run `uv run pytest` before each implementation commit.

Acceptance criteria:

- The normal suite runs without Docker, network access, or an OpenAI key.
- Opt-in integration tests verify the real Qdrant boundary.
- The chosen defaults are justified by recorded evaluation results rather than intuition alone.

## Phase 10 - Documentation and operational handoff

- [ ] Update the README with the final architecture and the separation between scraping, export, indexing, and chat.
- [ ] Document the complete local workflow:
  1. build the YouTube dataset
  2. export JSONL
  3. start Qdrant
  4. run the indexer
  5. launch the web dashboard
- [ ] Document environment variables, model configuration, expected API costs, and key rotation practices.
- [ ] Document how to inspect Qdrant point counts and verify the active embedding model.
- [ ] Document safe collection rebuild and backup procedures.
- [ ] Add troubleshooting for missing captions, malformed JSONL, embedding rate limits, Qdrant connectivity, dimension mismatch, and empty retrieval results.
- [ ] Document current limitations: YouTube-only corpus, all-video search, no authentication, no streaming, and no automatic stale-point deletion.
- [ ] Ensure every code/configuration change is committed as required by `AGENTS.md`.

Acceptance criteria:

- A new developer can follow the README from a clean checkout to a cited answer.
- No undocumented manual database changes are required.

## Recommended implementation order

1. Baseline tests and dependency approval.
2. Configuration and Docker-hosted Qdrant.
3. JSONL schema validation and the idempotent indexer.
4. Retrieval service and diagnostics.
5. Grounded OpenAI answer generation with citation validation.
6. Persistent SQLite conversation storage.
7. RAG API endpoints.
8. Web chat interface.
9. Evaluation, hardening, and documentation.

## Deferred features

- Streaming assistant responses.
- Automatic pruning of Qdrant points missing from the latest JSONL export.
- Video-specific retrieval filters.
- PDF, Markdown, text, or website ingestion.
- Multiple users, authentication, and per-user collections.
- Hosted Qdrant or production cloud deployment.
- Hybrid keyword/vector search and reranking.
- Automated background re-indexing after dataset export.
