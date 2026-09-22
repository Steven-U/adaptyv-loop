# Sovereign biology evolution

This repo remains the auditable V1 baseline: method history, per-design score,
budget guardrails, experiment submission, and result collection.

The next layer adds experimental state without replacing that baseline.

## M1 now implemented

The context module records target, assay, protocol version, model version,
reagent lot, instrument, timestamp, and observation kind.

Historical evidence is weighted by context similarity and recency. QC and
technical failures remain observable events but do not silently become
biological non-binders.

Campaign.plan_round can accept externally built posteriors, so a Sovereign
world-state layer can drive selection while the existing spend and execution
guardrails remain unchanged.

## Integrity fix

The README previously said the literal unknown method bucket was reported but
never allocated to. The selector did in fact learn a posterior for it. Because
that bucket has a high observed hit rate in the EGFR dataset, it can bias the
warm-start benchmark.

The implementation now keeps unknown provenance at the prior and does not
reserve an exploration slot for it. Rerun the original backtest before quoting
the historical 13-40% warm-start lift as a current result.

## Benchmark sequence

B0: rerun the original EGFR benchmark after the integrity fix.

B1: temporal regime shift. Compare context-free, recency-only, and
context-plus-recency posteriors after a protocol, assay, or model transition.

B2: failure semantics. Mix biological negatives with QC, expression, and
technical failures and measure posterior calibration and selection regret.

B3: dependency invalidation. Represent hypothesis -> candidate family ->
planned experiment, then measure avoided spend when new evidence invalidates
downstream work before execution.

B4: information value. Add actions where the most likely immediate binder is
not the most informative experiment. Measure cumulative multi-round utility.

B5: independent biological transfer. Only use a second real target/campaign to
make a cross-campaign transfer claim.

The intended end state is not merely a better protein ranker. It is an
auditable experimental world state that can explain why new evidence changed
what should be tested, cancelled, analyzed, or escalated next.
