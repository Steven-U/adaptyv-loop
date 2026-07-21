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

## What the data actually says

Everything below is measured on Adaptyv's public EGFR competition results
([round 1](https://github.com/adaptyvbio/egfr_competition_1),
[round 2](https://github.com/adaptyvbio/egfr_competition_2), ODbL) — 402
designs carrying both the computational metrics you have *before* spending
money and the wet-lab outcome. Reproduce with `python backtest/run_backtest.py`.

**The standard filters barely discriminate.** Binder/non-binder AUC, where
0.5 is a coin flip:

| metric | AUC | |
|---|---|---|
| pLDDT | 0.656 | weak but real |
| ipTM | 0.636 | weak but real |
| ESM2 pseudo-LL | 0.547 | essentially uninformative |
| `pae_interaction` | 0.388 | points the **wrong way** vs. the usual "lower iPAE is better" |

**Design method is the real lever, and it's free to act on.** Hit rate by
declared method, 14.0% base rate:

| method | n | binders | rate |
|---|---|---|---|
| ProteinMPNN/LigandMPNN | 30 | 13 | 43.3% |
| Custom PLM | 57 | 14 | 24.6% |
| BindCraft | 49 | 6 | 12.2% |
| ProteinMPNN/LigandMPNN + RFdiffusion | 83 | 6 | 7.2% |
| Custom ensemble/diffusion | 41 | 2 | 4.9% |
| Rosetta | 6 | 0 | 0.0% |

A 9x spread across methods, against a 0.64 AUC on individual designs. The most
popular method in the competition was also nearly the worst.

**A learned per-design model does not survive an honest transfer test.**
Cross-validated on round 2 it looks great — 25.1 binders per 80 tests against
ipTM's 19.3. Trained on round 1 and applied to round 2, which is the setup you
actually face, it drops to 16.0 and *loses* to plain ipTM. That is why this
package ships no learned per-design model.

**The baseline is weaker than it looks.** ipTM is quantized to two decimals, so
78 distinct values cover 378 designs and every budget cut lands inside a large
tie group. Breaking ties by row order instead of at random swings the ipTM
baseline between 9 and 17 binders at K=60 on luck alone. All comparisons here
randomize ties for every strategy.

## Does the policy help?

Two honest answers, because the result depends on whether you have history.

**Cold start — no prior campaign. It matches ipTM, it does not beat it.**

| strategy | binders (60 tests, $5,940) | cost/binder |
|---|---|---|
| random draw | 8.3 | $714 |
| rank by ipTM | 15.8 | $377 |
| adaptyv-loop | 15.6 | $381 |

Two rounds isn't enough to learn method rates, and on this pool ipTM already
acts as a proxy for method — its top 60 designs are 23 ProteinMPNN entries
supplying 13 of the 17 binders found there. Claiming a cold-start win would be
overfitting to one dataset.

**Warm start — method rates carried over from one prior campaign. +20–40%.**
Designs split in half at random; half A is the prior campaign, half B is the
new pool. No design appears in both, only the method rates transfer.

| strategy | 20 tests | 40 tests | 60 tests |
|---|---|---|---|
| random draw | 2.9 | 5.6 | 8.3 |
| rank by ipTM | 5.5 | 9.7 | 13.0 |
| method history | 6.9 (+26%) | 11.6 (+20%) | 15.7 (+20%) |
| method history + ipTM | **7.7 (+41%)** | **11.6 (+19%)** | 14.7 (+13%) |

Worth roughly $100–150 per binder found. The value is in *keeping the history*,
not in any single ordering — which is the argument for wiring the loop together
rather than exporting a CSV per campaign.

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
at the bench. Three rounds of 45 designs on a $14k budget reach a 20–22% hit
rate against the 14.0% base rate, around $460–495/binder, then round four is
refused for want of budget. Re-running resumes at round four rather than
restarting. The spread across runs is tie-breaking, which is randomized by
design.

This one is offline because a live learning loop is undemoable: each round
bills ~$4,455 and takes about three weeks, so three rounds is $13k and two
months.

**`python demo_live.py --target EGFR`** — against the real API, spending
nothing. Uses only non-billable endpoints: `/whoami`, `/targets`, and
`/experiments/cost-estimate`. It authenticates, resolves a real catalog target,
selects designs under budget, asks Foundry what the experiment would actually
cost, then shows the guardrails passing and deliberately failing. The submit
step is printed rather than executed unless you pass
`--i-want-to-spend-money`.

## Install

```bash
pip install -e .            # runtime: requests
pip install -e '.[dev]'     # + pytest, pandas, numpy, scikit-learn
pytest                      # 56 tests
python backtest/run_backtest.py
```

## Layout

| path | |
|---|---|
| `adaptyv_loop/client.py` | Foundry API client — auth, retries, backoff, pagination, spec validation |
| `adaptyv_loop/guardrails.py` | spend ceilings, dry-run gate, append-only decision log |
| `adaptyv_loop/selection.py` | Beta-Binomial method posteriors, budget allocation |
| `adaptyv_loop/campaign.py` | the resumable loop |
| `adaptyv_loop/report.py` | spend and yield in budget-holder units |
| `backtest/run_backtest.py` | all four analyses above, reproducible |

## Caveats

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
  untested against the live API.

Data: Adaptyv EGFR competition rounds 1–2, ODbL. Not affiliated with Adaptyv Bio.
