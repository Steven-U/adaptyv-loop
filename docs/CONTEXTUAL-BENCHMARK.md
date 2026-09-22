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
