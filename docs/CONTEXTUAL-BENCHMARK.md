# Context-aware benchmark

The original loop is the baseline: design-method history, per-design score,
budget controls, experiment submission, and result collection.

This extension asks one narrow question: when experimental conditions change,
should old evidence count exactly as much as recent evidence collected under the
current conditions?

The context layer records target, assay, protocol version, model version,
reagent lot, instrument, timestamp, and whether an observation is biological or
a QC/technical failure. Historical evidence can then be discounted by recency
and context mismatch.

Two benchmarks are kept separate:

- **B0 — baseline revalidation.** Rerun the original EGFR retrospective after
  the reporting-only `unknown` method bucket was fixed so it cannot receive a
  learned allocation advantage.
- **B1 — controlled regime shift.** Use a synthetic two-method environment
  where a protocol change reverses which method performs best. Compare
  context-free history, recency-only history, and context + recency weighting.

B1 is a software benchmark, not biological evidence. It tests whether the
mechanism behaves correctly when relevance of historical evidence changes.


## B1 observed result

With 400 deterministic seeds, 120 old observations per method, and six rounds
of 20 post-change tests:

| planner | binders / 120 | hit rate | stale-regime tests | mean recovery round |
|---|---:|---:|---:|---:|
| context-free | 12.0 | 10.0% | 120.0 | 7.00 |
| recency-only | 12.2 | 10.1% | 119.5 | 6.97 |
| context + recency | 36.0 | 30.0% | 50.8 | 3.54 |

The +200.4% binder difference is intentionally **not** a biological claim. The
benchmark creates a sharp protocol regime change where the previously best
method becomes the worst, so it tests whether stale evidence can be discounted
when the relevant context changes.
