# TASK-005 — Add Qdrant and explicit application connectivity

Status: Groomed

## Goal

Qdrant runs as a durable internal Compose service, and the application uses
the configured server and collection. A failed external connection is an
explicit startup/retrieval error rather than an accidental switch to an empty
in-memory database.

## Acceptance criteria

- [ ] The `qdrant` service uses a pinned image tag, mounts a named storage
      volume, and has an HTTP health check that succeeds for that image.
- [ ] Qdrant's REST and gRPC ports are internal by default and are not
      published to the host by the Compose file.
- [ ] The default application retrieval path reads `QDRANT_URL` and
      `QDRANT_COLLECTION`; with the example environment it targets
      `http://qdrant:6333` and `tw3k_transcripts`.
- [ ] A reachable configured Qdrant server is used for collection creation,
      indexing, and search; the client does not silently substitute another
      server.
- [ ] When the configured server is unavailable, construction or the first
      required operation raises a clear connection error and emits no
      successful in-memory fallback behavior.
- [ ] In-memory Qdrant remains available only when a caller explicitly opts in
      (for example, a test double or an intentional local mode); the default
      application and Compose paths never opt in implicitly.

## Out of scope

- [ ] Configurable ingestion, collection population, force reindexing, and
      the `qdrant-init` job, moved to
      [TASK-007](007-qdrant-indexing.md).
- [ ] Model-volume preparation and the application/volume dependency graph,
      moved to [TASK-006](006-model-preparation.md) and
      [TASK-008](008-compose-application.md).
- [ ] Operator documentation, unit tests, and full Compose lifecycle checks,
      moved to [TASK-009](009-operator-documentation.md),
      [TASK-010](010-container-tests.md), and
      [TASK-011](011-compose-lifecycle.md).

## Constraints

- Use the existing `qdrant-client` integration; do not add another vector
  database or dependency without approval.
- Never use `localhost` for the Qdrant host inside the Compose service graph.
- Preserve explicit in-memory support for isolated tests while making failure
  of the configured external service observable.
- Keep the collection name and URL overridable through the environment
  contract rather than hard-coding Compose-only values in reusable classes.
