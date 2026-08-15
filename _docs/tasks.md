# Local TW3K RAG Assistant Tasks

## Goal

Build a local retrieval-augmented generation (RAG) assistant that indexes the
existing TW3K JSONL transcript dataset in Qdrant, retrieves relevant guidance,
and uses OpenAI to generate answers grounded only in the retrieved context.

```text
tw3k_dataset.jsonl
  -> sentence-transformers embeddings
  -> local Qdrant
  -> retrieve and rerank
  -> OpenAI grounded answer
  -> local web interface with timestamp citations
```

## Confirmed product decisions

- Run the application locally for a single user with no authentication.
- Run Qdrant through Docker Compose with persistent local storage.
- Use `sentence-transformers/all-MiniLM-L6-v2` for local dense embeddings.
- Configure the Qdrant collection for 384-dimensional cosine-distance vectors.
- Use `cross-encoder/ms-marco-MiniLM-L6-v2` to rerank retrieved candidates.
- Use OpenAI only for answer generation, not embeddings.
- Keep the OpenAI generation model configurable rather than hard-coded.
- Search all indexed content without campaign-faction classification or filters.
- Treat campaign transcripts as sources of generally applicable game guidance.
- Never substitute unrelated guidance when relevant evidence is unavailable.
- Return answers with clickable links to the source YouTube timestamps.
- Prioritize retrieval quality over minimizing local computation.

## Dataset facts

The current `tw3k_dataset.jsonl` is an example dataset that will grow later.
At the time this plan was written, it contains:

- 2,655 transcript chunks from 22 videos
- approximately 25.9 hours of video
- approximately 340,634 estimated tokens, including chunk overlap
- stable chunk IDs, video IDs, chunk indexes, transcript text, timestamps, video
  metadata, and timestamp links
- no missing fields or duplicate chunk IDs

The current chunks are already an appropriate embedding size. Their deliberate
overlap must be handled during context selection so repeated text is not sent to
OpenAI.

## Phase 0 - Baseline and dependency approval

- [ ] Run the existing test suite and record the baseline result.
- [ ] Confirm the active application directory and existing project structure.
- [ ] Ask for approval before adding dependencies, as required by `AGENTS.md`.
- [ ] Add approved dependencies only to `pyproject.toml` and update `uv.lock` with
  `uv sync`.
- [ ] Expected dependencies include the Qdrant client, sentence-transformers,
  and the OpenAI Python SDK; reuse existing web dependencies where possible.
- [ ] Add an example environment file containing variable names but no secrets.

Acceptance criteria:

- The pre-existing test suite passes or any existing failures are documented.
- Installation is reproducible with `uv sync`.
- No API key or generated Qdrant data is committed.

## Phase 1 - Local Qdrant and configuration

- [ ] Add Docker Compose configuration for Qdrant.
- [ ] Persist Qdrant data in a named Docker volume.
- [ ] Add a Qdrant health check.
- [ ] Document commands for starting, stopping, and inspecting Qdrant.
- [ ] Add centralized configuration for:
  - `OPENAI_API_KEY`
  - OpenAI generation model
  - embedding model
  - reranker model
  - Qdrant URL and collection name
  - initial retrieval count
  - final context count and context budget
  - calibrated relevance threshold
- [ ] Fail with clear messages when required configuration is missing.

Acceptance criteria:

- Qdrant starts with one documented Docker Compose command.
- Indexed vectors survive container restarts.
- Application errors distinguish missing configuration from unavailable Qdrant.

## Phase 2 - JSONL validation and indexing contract

- [ ] Validate every JSONL record before indexing it.
- [ ] Report malformed data with its JSONL line number and reason.
- [ ] Require stable `chunk_id`, `video_id`, `chunk_index`, and non-empty `text`.
- [ ] Preserve all existing source and citation fields in the Qdrant payload.
- [ ] Construct embedding input from `video_title` and `text`.
- [ ] Preserve the original transcript text separately for answers and citations.
- [ ] Define deterministic Qdrant point IDs derived from stable chunk IDs.
- [ ] Record the embedding model and vector dimensions in index metadata.
- [ ] Reject an existing collection with an incompatible vector configuration.

Qdrant payloads should retain at least:

- `chunk_id`
- `video_id`
- `chunk_index`
- `text`
- `start_time` and `end_time`
- `formatted_time`
- `timestamp_link`
- `video_title`
- `channel`
- `video_url`

Campaign-faction metadata is intentionally out of scope.

Acceptance criteria:

- Validation can run without Qdrant, sentence-transformers, or OpenAI access.
- Invalid input cannot silently produce a partially incompatible index.
- Every indexed point can be traced back to its JSONL record and timestamp link.

## Phase 3 - Idempotent Qdrant indexer

- [ ] Build a standalone command that accepts a JSONL path.
- [ ] Load `sentence-transformers/all-MiniLM-L6-v2` locally.
- [ ] Create 384-dimensional embeddings in batches.
- [ ] Create the Qdrant collection with cosine distance when it does not exist.
- [ ] Upsert records in batches using stable point IDs.
- [ ] Make rerunning the indexer update records without creating duplicates.
- [ ] Read large JSONL files incrementally.
- [ ] Check Qdrant connectivity before beginning embedding work.
- [ ] Report validated, embedded, upserted, skipped, and failed record counts.
- [ ] Return a nonzero exit code for incomplete indexing.
- [ ] Add a validation-only or dry-run option.

Acceptance criteria:

- Indexing the same file twice leaves the same point count.
- Editing one chunk and reindexing updates its vector and payload.
- Adding one chunk and reindexing increases the point count by one.
- The current 2,655-record dataset can be indexed locally without OpenAI calls.

## Phase 4 - Retrieval and reranking

- [ ] Embed each question using the same MiniLM model as the indexer.
- [ ] Retrieve an initial configurable candidate set from Qdrant, starting at 20.
- [ ] Rerank candidates using
  `cross-encoder/ms-marco-MiniLM-L6-v2`.
- [ ] Preserve retrieval and reranking scores for local diagnostics.
- [ ] Deduplicate overlapping transcript chunks.
- [ ] Join adjacent chunks from the same video when they complete an explanation.
- [ ] Select approximately 6-10 final passages within a configurable context
  budget.
- [ ] Calibrate an evidence-sufficiency threshold with evaluation questions.
- [ ] Return a deterministic insufficient-evidence result when retrieved context
  is clearly irrelevant or empty.
- [ ] Add a diagnostic command or endpoint that shows retrieved sources without
  calling OpenAI.

Acceptance criteria:

- Known questions retrieve their expected tutorial passages.
- Repeated overlap does not dominate the final context.
- Every selected passage retains a valid source link.
- Unsupported questions are stopped before answer generation when possible.

## Phase 5 - Grounded OpenAI answer generation

- [ ] Build a generation service that receives only the question and selected
  source context.
- [ ] Keep the OpenAI model configurable.
- [ ] Give each supplied source an application-controlled citation identifier.
- [ ] Require the model to answer only from supplied context.
- [ ] Require the model to avoid outside knowledge and unsupported
  faction-specific claims.
- [ ] Require the model to say that relevant information was not found when the
  context is insufficient.
- [ ] Require citations for substantive advice.
- [ ] Validate that every returned citation refers to a supplied source.
- [ ] Render citations as links to each record's `timestamp_link`.
- [ ] Never turn an OpenAI or Qdrant failure into a fabricated answer.

The answer instruction must enforce this contract:

> Answer exclusively from the supplied context. Do not use outside knowledge or
> infer faction-specific details that the sources do not support. If the context
> does not contain enough relevant information, state that relevant information
> was not found in the available sources. Cite the supplied sources for every
> substantive recommendation.

Acceptance criteria:

- Generated claims are supported by supplied passages.
- Citations open the correct video at the relevant timestamp.
- Invalid model-generated citation IDs are rejected or repaired.
- Insufficient evidence produces an explicit, honest response.

## Phase 6 - Local API and web interface

- [ ] Add a FastAPI endpoint for submitting a question.
- [ ] Add readiness checks for the application, Qdrant, loaded local models, and
  OpenAI configuration.
- [ ] Keep model and network work outside database transactions.
- [ ] Add a local web page with a question input and generated answer area.
- [ ] Show clickable timestamp citations and expandable source excerpts.
- [ ] Show clear loading, empty-index, insufficient-evidence, Qdrant-error, and
  OpenAI-error states.
- [ ] Prevent duplicate submissions.
- [ ] Sanitize transcript and generated content before rendering it.
- [ ] Make the interface usable on desktop and mobile.

Acceptance criteria:

- A local user can ask a question and receive a grounded, cited answer.
- Clicking a citation opens its exact YouTube timestamp.
- The interface clearly distinguishes no evidence from service failures.

## Phase 7 - Testing and retrieval evaluation

- [ ] Add unit tests for JSONL validation, stable IDs, embedding batching,
  collection compatibility, deduplication, context selection, and citations.
- [ ] Mock Qdrant and OpenAI in the default suite so it is deterministic and
  does not incur API costs.
- [ ] Add an opt-in Docker integration test for Qdrant create, upsert, update,
  and search behavior.
- [ ] Create a small synthetic JSONL fixture for automated tests.
- [ ] Create 30-50 representative evaluation questions covering early-game
  priorities, economy, food, armies, diplomacy, reforms, corruption, battles,
  specific names, and unsupported requests.
- [ ] Record retrieval hit rate, citation validity, groundedness, and latency.
- [ ] Tune candidate count, reranking, relevance thresholds, neighboring-chunk
  expansion, and context size using evaluation results.
- [ ] Add regression cases for irrelevant questions and prompt injection inside
  transcript text.
- [ ] Run `uv run pytest` before each implementation commit.

Acceptance criteria:

- Normal tests require neither Docker nor an OpenAI API key.
- Real-Qdrant tests are available as an explicit opt-in suite.
- Retrieval defaults are supported by recorded evaluation results.

## Recommended implementation order

1. Baseline tests and dependency approval.
2. Docker Compose Qdrant and application configuration.
3. JSONL validation and idempotent local embedding indexer.
4. Dense retrieval, reranking, overlap handling, and evidence gating.
5. Grounded OpenAI generation with validated citations.
6. Local FastAPI endpoint and web interface.
7. Retrieval evaluation, tuning, documentation, and hardening.

## Deferred features

- Campaign-faction classification and metadata filtering
- Hybrid dense and keyword retrieval
- Automatic deletion of stale Qdrant points missing from a new JSONL export
- Hosted Qdrant or cloud deployment
- Authentication and multiple users
- Additional document types beyond the current JSONL transcript contract
- Automatic background indexing
- Persistent multi-conversation chat history
