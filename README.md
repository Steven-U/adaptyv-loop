# adaptyv-loop

A budgeted design–test–learn loop over the [Adaptyv Foundry API](https://docs.adaptyvbio.com/).

Protein design teams generate designs far faster than they can test them. At
$99/protein a 400-design campaign runs about $40k, and in Adaptyv's own public
competition data roughly 86% of that bought non-binders. This package closes
the loop: it decides which designs to buy under a budget, submits them, pulls
results back, and folds those results into the next round's decision.

It also puts three controls between an automated loop and a real invoice,
because the API's `auto_accept_quote` flag will finalize a Stripe quote and
create an invoice with no human in the loop.

```python
from adaptyv_loop import Campaign, Candidate, FoundryClient, SpendPolicy

client = FoundryClient()                       # reads ADAPTYV_API_TOKEN
campaign = Campaign(
    client,
    campaign_id="egfr-q3",
    target_id=egfr_id,
    policy=SpendPolicy(max_experiment_usd=5_000, max_campaign_usd=20_000),
)

selection = campaign.plan_round(candidates, n_slots=48)
campaign.dry_run(selection)                    # free: prices and validates
record = campaign.submit_round(selection)      # bills, only after guardrails pass
campaign.collect_round(record, candidates)     # results feed the next round
```

## The result

**Allocating test budget by design method finds 13–40% more binders per dollar
than ranking by ipTM** — the standard filter — once you have one campaign of
history. Method-history allocation holds a steady +20–27% across every budget
tested; adding ipTM as a tiebreaker is stronger at small budgets (+40% at 20
tests) and weaker at larger ones (+13% at 60). At these budgets that's worth
roughly $100–150 per binder found.

The reason is a signal most pipelines throw away. Everyone filters on
per-design confidence scores, which turn out to be weak, while the design
*method* separates outcomes by 9x and is knowable before a dollar is spent.
This package makes method the primary allocation unit and keeps the history
that makes it work.

Everything below is measured on Adaptyv's public EGFR competition results
([round 1](https://github.com/adaptyvbio/egfr_competition_1),
[round 2](https://github.com/adaptyvbio/egfr_competition_2), ODbL) — 402
designs carrying both the computational metrics you have *before* spending
money and the wet-lab outcome. Reproduce with `python backtest/run_backtest.py`.

**Design method is the lever.** Hit rate by declared method, 14.0% base rate,
all 15 categories as the backtest prints them:

| method | n | binders | rate |
|---|---|---|---|
| ProteinMPNN/LigandMPNN | 30 | 13 | 43.3% |
| *unknown* | 13 | 5 | *38.5%* |
| Custom PLM | 57 | 14 | 24.6% |
| Custom generative | 11 | 2 | 18.2% |
| BindCraft | 49 | 6 | 12.2% |
| ESM2/3 + ProteinMPNN/LigandMPNN + Rosetta | 9 | 1 | 11.1% |
| ESM2/3 + Rosetta | 10 | 1 | 10.0% |
| AlphaFold2 | 13 | 1 | 7.7% |
| TIMED | 13 | 1 | 7.7% |
| ProteinMPNN/LigandMPNN + RFdiffusion | 83 | 6 | 7.2% |
| ESM2/3 | 16 | 1 | 6.2% |
| Custom ensemble/diffusion | 41 | 2 | 4.9% |
| Custom generative + Custom surrogate | 5 | 0 | 0.0% |
| ESM2/3 + EvoProtGrad | 8 | 0 | 0.0% |
| Rosetta | 6 | 0 | 0.0% |

`unknown` posts the second-highest rate in the table and is deliberately not
allocated to: it is an absence of a declared method, not a method, so there is
nothing to buy more of. It is shown rather than dropped because excluding a
38.5% row from a table arguing that method predicts binding would be exactly
the kind of quiet selection this repo is trying not to do. Nine of the fifteen
categories carry n < 20, which is why the selection policy shrinks every
method's posterior toward the base rate rather than trusting these rates
directly.

The most popular method in the competition was also nearly the worst: 83
designs used RFdiffusion + ProteinMPNN for a 7.2% return, while 30 designs
using ProteinMPNN/LigandMPNN alone returned 43.3%.

**The per-design scores everyone filters on are much weaker.** Binder /
non-binder AUC, where 0.5 is a coin flip:

| metric | AUC | |
|---|---|---|
| pLDDT | 0.656 | weak but real |
| ipTM | 0.636 | weak but real |
| ESM2 pseudo-LL | 0.547 | essentially uninformative |
| `pae_interaction` | 0.388 | points the **wrong way** vs. the usual "lower iPAE is better" |

A 9x spread across methods against a 0.64 AUC on individual designs. So method
drives the allocation, and the per-design score only orders candidates once the
budget is split.

## Measured lift

Method rates carried over from one prior campaign. Designs are split in half at
random: half A is the prior campaign whose results are known, half B is the new
pool being selected from. No design appears in both — only the method rates
transfer.

| strategy | 20 tests | 40 tests | 60 tests |
|---|---|---|---|
| random draw | 2.8 | 5.5 | 8.4 |
| rank by ipTM | 5.5 | 9.6 | 12.9 |
| method history | 6.9 (+27%) | 11.6 (+20%) | **15.5 (+20%)** |
| method history + ipTM | **7.7 (+40%)** | 11.5 (+19%) | 14.5 (+13%) |

Worth roughly $100–150 per binder found. The value is in *keeping the history*,
not in any single ordering — which is the argument for wiring the loop together
rather than exporting a CSV per campaign.

Averaged over 400 seeds. The run is deterministic: seeds are `range(--trials)`,
so `python backtest/run_backtest.py` reproduces this table exactly.

## How selection works

Each candidate is scored as `method_rate × score_lift(global score percentile)`.

- **Method rate** is a Beta-Binomial posterior over that method's hit rate,
  shrunk toward a reference base rate. Shrinkage is what stops a method that
  went 3-for-3 in a pilot from eating the whole budget; it posts a posterior
  near 0.35, not 1.0.
- **Score lift** is calibrated, not assumed: the top 16% of the pool by ipTM
  hit at 28.3% against a 14.0% base, so the top of the ranking is worth about
  2x the average design. Not 10x.
- **Scores are ranked globally**, not within method, because ipTM is largely a
  proxy for method and ranking within method discards exactly the cross-method
  ordering that makes it useful before any history exists.
- **`max_method_fraction` is a safety rail, not a diversification mandate.**
  The data argues against diversifying — binders concentrate heavily in one
  method, and an aggressive cap spends budget moving away from it.
- **Ties break at random**, so a caller cannot accidentally profit from how
  their dataframe happened to be sorted.

`mode="explore"` Thompson-samples methods that already have observations;
`mode="exploit"` uses posterior means. `Campaign.plan_round` switches from
explore to exploit at 30 observations.

## Guardrails

The API can create an invoice with no human step. Three controls sit in front
of that:

1. **Spend ceilings**, per experiment and per campaign, enforced against
   Foundry's own `cost-estimate` rather than our guess at the price.
2. **A dry-run gate.** Every submission must first pass a free rehearsal.
   The gate is keyed on a fingerprint of the exact spec, so editing the
   sequence list after pricing invalidates the approval.
3. **An append-only decision log**, flushed and `fsync`ed *before* the billable
   call, so a crash mid-submission leaves a record rather than silence.

Budget state is reconstructed from the log on startup, so a restarted campaign
resumes its budget instead of getting a fresh one. `auto_accept_quote` is
refused unless the policy explicitly arms it.

```
GUARDRAIL REFUSED THE ROUND: $4,455.00 exceeds remaining campaign budget
of $635.00 (spent $13,365.00 of $14,000.00)
Nothing was submitted and no invoice exists.
```

## Demos

**`python demo_offline.py`** — the full loop, no token, no spend. The
simulated lab answers with the *real* wet-lab outcomes from the EGFR
competition, so every posterior update the loop learns from actually happened
at the bench. Three rounds of 45 designs on a $14k budget reach a 19–22% hit
rate against the 14.0% base rate, around $445–515/binder, then round four is
refused for want of budget. Re-running resumes at round four rather than
restarting. The spread across runs is tie-breaking, which is randomized by
design.

This one is offline because a live learning loop is undemoable: each round
bills ~$4,455 and takes about three weeks, so three rounds is $13k and two
months.

**`./mock/serve.sh` then `python demo_mock.py`** — the full loop over real
HTTP, validated against Adaptyv's genuine published OpenAPI contract. No token,
no access, no spend.

```
demo_mock.py -> Prism (:4010) -> mock Foundry (:4011)
                  |                 |
                  |                 real EGFR target, $99/protein pricing,
                  |                 real wet-lab binding outcomes
                  |
                  validates every request AND response against
                  backtest/foundry_openapi.json — Adaptyv's real spec.
                  A non-conforming payload is a 4xx, not a silent pass.
```

This is the strongest correctness evidence in the repo, because the contract
check is adversarial: it found three real shape bugs while being built. The
`create` response keys the id as `experiment_id`, not `id` — reading `id`
silently yields `None` and the campaign then submits against a null
experiment. Results come back in the paginated list envelope rather than a
bare array. And `TargetPricing` is a tagged union whose `per_sequence` branch
requires `price_per_sequence_cents`, not the flat field I first assumed. All
three are now regression-tested.

It also shows the division of labour: Prism enforces the structural schema
(a spec missing `experiment_type` gets a 422), while the type/field matrix —
"thermostability rejects a `target_id`" — is documented only in prose in the
spec, so `build_experiment_spec` enforces that locally before a request is
ever sent.

**`python demo_end_to_end.py`** — the whole pipeline, design through decision.

```bash
python demo_end_to_end.py --build-library   # generate + score with ESM-2
./mock/serve.sh --library                   # terminal 1
python demo_end_to_end.py                   # terminal 2
```

Four generators propose variants of a real anti-EGFR nanobody scaffold — CDR
mutagenesis, ESM-2-guided design, germline recombination with a randomized
CDR3, and a uniform-random negative control — with CDRs located by conserved
framework anchors rather than fixed indices. ESM-2 scores them locally (about
14 seconds for 600 designs on an M4). Then the loop allocates, submits through
the contract-validated stack, and learns.

The bench for these designs is simulated and labelled as such: they are novel,
so no measurement for them exists anywhere. It sees only the sequence, never
the generator label, and its effect sizes are calibrated to the competition
data — 14% base rate solved by bisection, within-method AUC 0.64 against the
measured 0.636. Beyond ESM-2 score it carries two real biophysical terms the
score misses: CDR3 length deviation from the parent scaffold, and CDR
liabilities (N-glycosylation sequons, NG deamidation, DG isomerization, free
cysteine) which language models score as perfectly ordinary.

**Result, reported as measured:**

| strategy | binders | hit rate | cost/binder |
|---|---|---|---|
| buy blind from the library | 22 | 14.0% | $707 |
| rank by ESM-2 score only | 43 | 26.9% | $368 |
| adaptyv-loop | 40 | 25.0% | $396 |

The loop **tied** score-ranking here (−1.9pp, 0.4 sigma), it did not beat it.
That is the expected result and it corroborates the main finding rather than
denting it: on this library ESM-2 PLL happens to rank all four generators in
exactly their true quality order, so allocating by method is redundant with
sorting by score. Method allocation pays only when method carries signal the
per-design score misses — the warm-start case on the real data. It is the same
mechanism that made the real cold-start case only match ipTM.

What the loop did add: it identified the dead generator without being told and
spent **zero** of its budget there, though that generator is 25% of the
library. And it does not depend on the score happening to be this well aligned.

**`python demo_live.py --target EGFR`** — the same flow against the real API
once you have a token, using only non-billable endpoints. Foundry tokens
require onboarding through an organization account, so this one is here for
completeness rather than as the demo.

## Install

```bash
pip install -e .            # runtime: requests
pip install -e '.[dev]'     # + pytest, pandas, numpy, scikit-learn
pip install -e '.[design]'  # + torch, fair-esm (generation and scoring)
pytest                      # 76 tests
python backtest/run_backtest.py   # needs '.[dev]' — section 3 skips without scikit-learn
```

## Layout

| path | |
|---|---|
| `adaptyv_loop/client.py` | Foundry API client — auth, retries, backoff, pagination, spec validation |
| `adaptyv_loop/guardrails.py` | spend ceilings, dry-run gate, append-only decision log |
| `adaptyv_loop/selection.py` | Beta-Binomial method posteriors, budget allocation |
| `adaptyv_loop/campaign.py` | the resumable loop |
| `adaptyv_loop/design.py` | four generators + local ESM-2 scoring |
| `adaptyv_loop/bench.py` | simulated bench for novel designs, calibrated to the real data |
| `adaptyv_loop/report.py` | spend and yield in budget-holder units |
| `backtest/run_backtest.py` | all four analyses above, reproducible |
| `mock/mock_foundry.py` | local Foundry serving real EGFR data in real schema shapes |
| `mock/serve.sh` | brings up the Prism-validated stack |
| `backtest/foundry_openapi.json` | Adaptyv's real published OpenAPI 3.1 spec |

## Method notes and limits

Three things worth knowing before quoting the numbers above.

**The lift needs history.** With no prior campaign, splitting a budget into an
explore round and an exploit round only *matches* ipTM ranking (15.5 vs 15.7
binders per 60 tests) rather than beating it. Two rounds isn't enough to learn
method rates, and on this pool ipTM is partly acting as a proxy for method
anyway — its top 60 designs are 23 ProteinMPNN entries supplying 13 of the 17
binders found there. The 13–40% figure is a warm-start number and is reported
as one.

**No learned per-design model, deliberately.** A gradient-boosted model on the
per-design metrics looks strong under cross-validation (25.1 binders per 80
tests vs ipTM's 19.3) but drops to 16.0 when trained on round 1 and applied to
round 2, which is the setup you actually face. Method-level allocation is where
the transferable signal lives, so that is what ships.

**Tie-breaking is load-bearing.** ipTM is quantized to two decimals, so 78
distinct values cover 378 designs and every budget cut lands inside a large tie
group. Ordering ties by dataframe row order instead of at random swings the
ipTM baseline between 9 and 17 binders at K=60 on luck alone. Every comparison
here randomizes ties for every strategy, including the baselines.

Remaining limits:

- Everything is measured on one target (EGFR) in a crowdsourced competition.
  The method mix in a competition is not the method mix inside one company's
  pipeline, and the warm-start result assumes method identity is recorded
  consistently across campaigns.
- The warm-start split shares a single campaign's population. It holds designs
  out, not campaigns, so it measures whether method rates generalize across
  designs — a genuine cross-campaign test needs a second target.
- `min_methods` and `max_method_fraction` were chosen for defensibility, not
  tuned to maximize the reported numbers. Tuning them on this dataset would
  make the reported lift optimistic.
- Only `screening` and `affinity` paths have been exercised end to end.
  Thermostability, expression, and enzyme-activity specs validate but are
  untested against a real Foundry instance.

## Licence

Code in this repository is Apache 2.0 (see `LICENSE`).

The competition data is **not redistributed here**. `backtest/data/` is
gitignored; `backtest/run_backtest.py` fetches it from Adaptyv's own
repositories at run time, where it is published under the
[ODbL](https://opendatacommons.org/licenses/odbl/). The measurements in this
README and in `docs/analysis-log.md` are a Produced Work derived from that
database, attributed above.

Data: Adaptyv EGFR competition rounds 1–2, ODbL. Not affiliated with Adaptyv Bio.
