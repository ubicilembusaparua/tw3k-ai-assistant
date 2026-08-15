# Retrieval evaluation

Recorded on 2026-08-15 against the 2,655-point local Qdrant index using
`sentence-transformers/all-MiniLM-L6-v2` and
`cross-encoder/ms-marco-MiniLM-L6-v2`.

The 40-question set contains 32 supported questions across game setup,
early-game priorities, economy, food, armies, diplomacy, reforms, corruption,
battles, and named characters, plus 8 unsupported requests. Run it with:

```powershell
uv run tw3k-evaluate
```

The full settings, metrics, per-question selections, scores, and timings are in
`results.json`. Metrics mean:

- Hit rate: a supported question selected a curated expected video or context
  containing a curated answer term.
- Citation validity: every selected link was an absolute HTTP(S) URL containing
  a timestamp parameter.
- Groundedness: selected context both hit the expected evidence and contained a
  curated answer term. This is a retrieval-context proxy; evaluation does not
  call OpenAI.
- Unsupported rejection: an unsupported question was stopped with
  `insufficient_evidence`.
- Latency: dense retrieval, cross-encoder reranking, gating, deduplication, and
  context selection wall time on the local evaluation machine.

## Recorded profile comparison

| Profile | Candidates | Final | Budget | Threshold | Neighbor | Hit | Unsupported | Grounded | Citations | Mean ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_neighbors | 10 | 6 | 3000 | 0.35 | 0 | 75.00% | 100% | 75.00% | 100% | 257.14 | 303.29 |
| **lean** | **10** | **6** | **3000** | **0.35** | **1** | **75.00%** | **100%** | **75.00%** | **100%** | **272.83** | **321.93** |
| balanced | 20 | 8 | 4000 | 0.35 | 1 | 75.00% | 100% | 71.88% | 100% | 511.57 | 558.98 |
| strict | 20 | 8 | 4000 | 1.0 | 1 | 71.88% | 100% | 68.75% | 100% | 466.39 | 503.07 |
| broad | 30 | 10 | 5000 | 0.0 | 1 | 78.12% | 87.50% | 78.12% | 100% | 690.96 | 741.97 |
| moderate | 20 | 8 | 4000 | -1.0 | 1 | 90.62% | 87.50% | 84.38% | 100% | 460.22 | 495.43 |
| moderate_broad | 30 | 10 | 5000 | -1.0 | 1 | 90.62% | 87.50% | 87.50% | 100% | 683.35 | 755.77 |
| permissive | 20 | 8 | 4000 | -2.0 | 1 | 90.62% | 87.50% | 84.38% | 100% | 456.28 | 490.51 |
| very_permissive | 30 | 10 | 5000 | -5.0 | 1 | 100% | 75.00% | 100% | 100% | 678.79 | 766.26 |

## Selected defaults

The `lean` profile is selected because unsupported rejection is a hard priority;
lower thresholds admitted unrelated or unsupported faction-specific requests.
Among profiles retaining one-chunk neighbor expansion and 100% unsupported
rejection, `lean` matched the best hit and groundedness rate with substantially
lower latency than 20- or 30-candidate alternatives.

- Candidate count: 10
- Final context count: 6
- Context budget: 3,000 approximate tokens
- Raw cross-encoder relevance threshold: 0.35
- Neighbor-chunk expansion: 1
- Overlap threshold: 0.75
