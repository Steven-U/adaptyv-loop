# Analysis log

Every test run while building this package, in order, with what it changed.
Entries are **KEEP** (shipped), **REJECT** (tried, failed, removed), or
**FIX** (a bug the test exposed). Failures are recorded because they are the
reason the design looks the way it does.

Data: Adaptyv's public EGFR competition results
([round 1](https://github.com/adaptyvbio/egfr_competition_1),
[round 2](https://github.com/adaptyvbio/egfr_competition_2), ODbL).
402 designs, 378 after dropping rows with missing metrics, 53 binders (14.0%).
Reproduce everything with `python backtest/run_backtest.py`.

Independent analysis. Not affiliated with Adaptyv Bio.

---

## 1. Do the standard filters predict binding? — **REJECT the premise**

Measured binder/non-binder AUC for every metric the dataset carries.

| metric | AUC | reading |
|---|---|---|
| pLDDT | 0.656 | weak but real |
| ipTM | 0.636 | weak but real |
| ESM2 pseudo-LL | 0.547 | essentially uninformative |
| `pae_interaction` | 0.388 | points the **wrong way** vs. the "lower iPAE is better" convention |

0.500 is a coin flip. Nothing here is a strong filter, and the one most
commonly used as a hard cutoff is inverted on this data.

**Consequence:** a campaign that filters on these is buying mostly
non-binders. At $99/protein and a 14% hit rate, roughly 86% of a $40k
400-design campaign buys nothing.

---

## 2. What does predict binding? — **KEEP: design method**

Highest and lowest of the 15 categories with n ≥ 5. The full table is in the
[README](../README.md#the-result) and in the backtest output.

| method | n | binders | rate |
|---|---|---|---|
| ProteinMPNN/LigandMPNN | 30 | 13 | 43.3% |
| *unknown* | 13 | 5 | *38.5%* |
| Custom PLM | 57 | 14 | 24.6% |
| BindCraft | 49 | 6 | 12.2% |
| ProteinMPNN/LigandMPNN + RFdiffusion | 83 | 6 | 7.2% |
| Custom ensemble/diffusion | 41 | 2 | 4.9% |
| Rosetta | 6 | 0 | 0.0% |

`unknown` is an absence of a declared method rather than a method, so it is
reported but never allocated to.

A 9x spread across methods against a 0.64 AUC on individual designs. The most
popular method in the competition (83 designs) was also nearly the worst.

**Consequence:** method becomes the primary allocation unit. This is the
finding the whole package is built on.

---

## 3. Can a model beat ipTM per design? — **REJECT**

Gradient-boosted classifier on the per-design metrics. Binders found per
budget:

| strategy | K=40 | K=80 | K=120 | K=200 |
|---|---|---|---|---|
| random (expected) | 5.6 | 11.2 | 16.8 | 28.0 |
| rank by ipTM | 11.0 | 19.3 | 26.0 | 34.2 |
| GBM, 5-fold CV on round 2 | 12.1 | **25.1** | 31.3 | 40.7 |
| GBM, trained round 1 → applied round 2 | 5.0 | **16.0** | 27.0 | 32.0 |

Cross-validation says the model wins by a wide margin. The honest setup —
train on the earlier campaign, select in the later one, which is what you
actually face — says it loses to plain ipTM.

**Consequence:** no learned per-design model ships. Had I stopped at the
cross-validated number I would have shipped something that degrades campaigns.

---

## 4. The baseline was lying — **FIX: randomize ties everywhere**

ipTM is quantized to two decimals, so **78 distinct values cover 378
designs** and every budget cut lands inside a large tie group. The top-20
cut falls inside a 22-design tie at 0.95.

Ranking by ipTM at K=60, depending only on how ties are ordered:

| tie handling | binders |
|---|---|
| pandas row order (`np.argsort`) | 17 |
| random tie-breaking, mean of 1000 draws | 15.7 |
| unlucky row order | 9 |

An 8-binder swing on nothing but input sort order. My first backtest had
accidentally manufactured a result this way.

**Consequence:** every comparison in the repo randomizes ties for every
strategy including baselines, and `_percentiles` gives tied values a shared
percentile so selection cannot depend on dataframe order.

---

## 5. First selection policy — **REJECT: diversification**

Beta-Binomial method posteriors, per-method concentration cap at 40% of
budget, forced coverage of ≥3 methods, gentle within-method score ordering.

| strategy | K=60 | K=100 | K=160 |
|---|---|---|---|
| rank by ipTM | 17.0 | 21.0 | 30.0 |
| policy v1 | **5.0** | 14.6 | 25.3 |

Worse than random at small budgets.

**Diagnosis:** ipTM's apparent strength is *method-confounded*. Its top 60
designs are 23 ProteinMPNN entries supplying 13 of the 17 binders found
there. So ipTM was already acting as a method proxy, and my diversity cap was
actively spending budget moving away from the one method that worked. Ranking
*within* method also discarded the cross-method ordering that made the metric
useful in the first place.

**Consequence:** the cap was reframed from a diversification mandate into a
safety rail against small-sample flukes (0.4 → 0.7), scores are ranked
globally rather than within method, and `score_lift` was recalibrated against
the data (top 16% of the pool hits at 28.3% vs a 14.0% base, so the top of the
ranking is worth ~2x, not 10x).

---

## 6. Zero-hit pilot destroys the prior — **FIX**

With a 20-design first round at a 14% base rate, finding zero binders is
common. `build_posteriors` was deriving its prior mean from the observed
pooled rate, so a 0-for-20 pilot set the prior to 0.0, flattened every
method's posterior to ~0.001, and made the next round's ranking arbitrary.

**Consequence:** the pooled estimate is shrunk toward a fixed reference base
rate with 20 pseudo-trials, so it can never collapse.

---

## 7. Does the fixed policy help? — **KEEP, with a stated boundary**

**Cold start**, no prior campaign, budget split explore/exploit:

| strategy | K=60 | K=160 |
|---|---|---|
| rank by ipTM | 15.7 | 30.0 |
| adaptyv-loop | 15.5 | 29.0 |

A tie. Two rounds is not enough to learn method rates, and ipTM already
proxies method on this pool.

**Warm start**, method rates carried from one prior campaign. Designs split in
half at random — half A is the prior campaign, half B is the new pool, no
design in both, only method rates transfer:

| strategy | 20 tests | 40 tests | 60 tests |
|---|---|---|---|
| random draw | 2.8 | 5.5 | 8.4 |
| rank by ipTM | 5.5 | 9.6 | 12.9 |
| method history | 6.9 (+26%) | 11.6 (+21%) | **14.4 (+12%)** |
| method history + ipTM | **8.5 (+56%)** | 11.2 (+17%) | 13.6 (+5%) |

**Revalidation note:** the original implementation accidentally learned a
posterior for the literal `unknown` method bucket. After fixing missing
provenance to remain at the prior, method-history-only still beats ipTM at all
three warm-start budgets (+12% to +26%). The score-combined policy is more
budget-sensitive (+5% to +56%). The value remains in keeping usable history,
but the exact lift is not a universal constant.

---

## 8. Contract validation against the real spec — **FIX: three client bugs**

The Foundry API needs org onboarding, so there is no token to test against.
Instead: their published OpenAPI 3.1 spec, served by a Prism proxy in front of
a local mock, validating every request and response.

Three real bugs in my own client, none of which unit tests would have caught,
because all three were wrong *assumptions about their response shapes*:

| bug | consequence |
|---|---|
| `CreateExpResponse` keys the id as `experiment_id`, not `id` | reading `id` yields `None`; the campaign then submits and polls a null experiment |
| `GET /experiments/{id}/results` returns the paginated envelope, not a bare array | results silently parse as empty |
| `TargetPricing` is a tagged union; `per_sequence` needs `price_per_sequence_cents` | pricing read as absent |

Also worth noting the division of labour: Prism enforces the structural schema
(a spec missing `experiment_type` gets a 422) but *not* the type/field matrix,
which exists only as prose in the spec — "thermostability rejects a
`target_id`" passes Prism and is caught locally by `build_experiment_spec`.
Neither layer alone is sufficient.

---

## 9. End-to-end: generate designs and buy the right ones — **TIE, reported as one**

Four generators over a real anti-EGFR nanobody scaffold, scored by local
ESM-2. 600 designs in ~14s on an M4.

| generator | mean ESM-2 PLL | true rate |
|---|---|---|
| plm_guided | −0.3135 | 24.2% |
| scaffold_mutagenesis | −0.3249 | 18.9% |
| germline_recombination | −0.3354 | 12.1% |
| random_control | −0.4543 | 0.9% |

Same budget, 160 tests, $15,840:

| strategy | binders | hit rate | cost/binder |
|---|---|---|---|
| buy blind from the library | 22 | 14.0% | $707 |
| rank by ESM-2 score only | 43 | 26.9% | $368 |
| adaptyv-loop | 40 | 25.0% | $396 |

Gap −1.9pp, standard error 4.9pp, 0.4 sigma. **A tie, not a win.**

### How that number survived three attempts to improve it

The first bench made outcome a pure function of ESM score. Score-ranking then
matched the loop *exactly* — which contradicted this repo's own headline
finding that method carries signal beyond per-design scores. That made the
simulator unfaithful, so I added terms the score genuinely misses, both real
biophysics:

1. **CDR3 length deviation** from the parent scaffold (r = +0.22 with PLL).
   Dropped `germline_recombination` from 16.0% to 10.7% true rate. Still a tie.
2. **CDR liabilities** — N-glycosylation sequons, NG deamidation, DG
   isomerization, free cysteine. These are documented developability red flags
   that language models score as perfectly ordinary dipeptides (measured
   correlation with PLL only −0.43). Still a tie.

Then I stopped. Continuing to add terms until the loop won would have been
exactly the overfitting rejected in §3.

**Why the tie is the right answer:** on this library ESM-2 PLL happens to rank
all four generators in *exactly* their true quality order, so allocating by
method is redundant with sorting by score. Method allocation pays only when
method carries signal the per-design score misses. That is the warm-start case
in §7, and it is the same mechanism that made the cold-start case there also
only match ipTM. Two independent setups, one condition.

**What the loop did add:** it identified the dead generator without being told
and spent **zero** of its budget there, though that generator is 25% of the
library.

---

## Standing rules this log produced

1. Never report a cross-validated number without the out-of-sample one (§3).
2. Randomize ties in any ranking comparison; check the unique-value count of
   the score first (§4).
3. A cap or constraint needs evidence it helps, not just a story (§5).
4. Validate the client against the vendor's published contract, not against
   assumptions about it (§8).
5. When a result refuses to appear after two principled attempts, report the
   null (§9).
