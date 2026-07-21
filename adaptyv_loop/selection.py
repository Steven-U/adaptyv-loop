"""Budget-aware selection of which designs to send to the lab.

Motivation, measured on Adaptyv's public EGFR competition data (402 designs
with wet-lab ground truth; see ``backtest/``):

* Per-design confidence metrics are weak binder/non-binder discriminators.
  ipTM scores AUC 0.648, pLDDT 0.656, ESM2 pseudo-log-likelihood 0.559, and
  pae_interaction 0.374 — the last one pointing the *opposite* way to the
  usual "lower iPAE is better" convention.
* The design *method* is a far stronger lever, and it is knowable before any
  money is spent. Hit rate ran from 43% (ProteinMPNN/LigandMPNN, 13/30) to
  7% (RFdiffusion + ProteinMPNN, 6/83) to 0% (Rosetta, 0/6).
* Fitting a model on those per-design metrics does not survive an honest
  transfer test. Trained on Round 1 and evaluated on Round 2 it lost to
  simply ranking by ipTM.

So this module does not try to out-predict ipTM per design. It allocates the
budget across methods using a Beta-Binomial posterior over each method's hit
rate, and only uses the per-design score to order candidates *within* a
method, where a weak signal is still better than none.

The two modes matter for a multi-round campaign. Round one has no history, so
``explore`` (Thompson sampling) spreads the budget and buys information. Later
rounds, once methods have separated, use ``exploit`` (posterior mean).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Sequence

Mode = Literal["explore", "exploit"]

UNKNOWN_METHOD = "unknown"

#: Pooled binder rate across Adaptyv's public EGFR competition data, used as
#: the prior mean for a method with no history of its own.
REFERENCE_BASE_RATE = 0.12
#: Pseudo-trials of :data:`REFERENCE_BASE_RATE` mixed into the pooled estimate.
GLOBAL_PRIOR_WEIGHT = 20.0


@dataclass(frozen=True)
class Candidate:
    """One design competing for a slot in the experiment."""

    id: str
    sequence: str
    #: Design method / model stack, e.g. "BindCraft" or "RFdiffusion+ProteinMPNN".
    method: str = UNKNOWN_METHOD
    #: Per-design confidence, higher is better. Normalize sign before passing
    #: (for iPAE, negate it).
    score: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodStats:
    """Observed outcomes for one design method."""

    method: str
    tested: int
    hits: int

    @property
    def raw_rate(self) -> float:
        return self.hits / self.tested if self.tested else 0.0


@dataclass
class MethodPosterior:
    """Beta posterior over a method's hit rate, shrunk toward the global base rate.

    Shrinkage is what stops a method that went 3-for-3 in a pilot from eating
    an entire campaign budget. ``prior_weight`` is the number of pseudo-trials
    of global-base-rate evidence mixed in; at the default of 8, a 3-for-3
    method with a 12% global rate posts a posterior mean near 0.35 rather
    than 1.0.
    """

    method: str
    alpha: float
    beta: float
    tested: int = 0
    hits: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def stdev(self) -> float:
        a, b = self.alpha, self.beta
        return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.alpha, self.beta)

    def credible_interval(self, width: float = 0.9) -> tuple[float, float]:
        """Normal approximation to the central credible interval.

        Adequate for the display and ranking this package does; not intended
        as a substitute for the exact Beta quantiles in a statistical report.
        """
        z = {0.5: 0.674, 0.8: 1.282, 0.9: 1.645, 0.95: 1.960}.get(width, 1.645)
        lo = max(0.0, self.mean - z * self.stdev)
        hi = min(1.0, self.mean + z * self.stdev)
        return lo, hi


def build_posteriors(
    history: Iterable[MethodStats],
    *,
    prior_weight: float = 8.0,
    global_rate: float | None = None,
) -> dict[str, MethodPosterior]:
    """Build per-method posteriors from observed results.

    ``global_rate`` defaults to the pooled hit rate across all history, which
    is the right prior mean for a method we have not seen yet.
    """
    stats = list(history)
    total_tested = sum(s.tested for s in stats)
    total_hits = sum(s.hits for s in stats)
    if global_rate is None:
        # Shrink the pooled estimate toward the reference rate rather than
        # adopting it outright. A 20-design pilot that happens to find zero
        # binders is common at a ~14% base rate, and taking its raw 0.0 as the
        # prior mean would flatten every method's posterior to zero and make
        # the next round's ranking arbitrary.
        global_rate = (total_hits + GLOBAL_PRIOR_WEIGHT * REFERENCE_BASE_RATE) / (
            total_tested + GLOBAL_PRIOR_WEIGHT
        )
    global_rate = min(max(global_rate, 1e-3), 1 - 1e-3)

    a0 = prior_weight * global_rate
    b0 = prior_weight * (1 - global_rate)

    out = {
        s.method: MethodPosterior(
            method=s.method,
            alpha=a0 + s.hits,
            beta=b0 + (s.tested - s.hits),
            tested=s.tested,
            hits=s.hits,
        )
        for s in stats
    }
    out.setdefault(
        UNKNOWN_METHOD,
        MethodPosterior(method=UNKNOWN_METHOD, alpha=a0, beta=b0),
    )
    return out


def _default_posterior(posteriors: Mapping[str, MethodPosterior], method: str) -> MethodPosterior:
    if method in posteriors:
        return posteriors[method]
    fallback = posteriors.get(UNKNOWN_METHOD)
    if fallback is not None:
        return MethodPosterior(method=method, alpha=fallback.alpha, beta=fallback.beta)
    return MethodPosterior(method=method, alpha=1.0, beta=7.0)


def _percentiles(values: Sequence[float | None]) -> list[float]:
    """Rank-percentile of each value, with missing scores placed at the median.

    Tied values share the average of the ranks they span. This matters more
    than it sounds: confidence metrics are routinely reported to two decimal
    places, so a 378-design pool can hold only 78 distinct ipTM values and a
    budget cut lands *inside* a tie group of twenty-odd designs. Assigning
    distinct percentiles within a tie would make the selection depend on input
    row order, which is not information.
    """
    present = sorted((v, i) for i, v in enumerate(values) if v is not None)
    out = [0.5] * len(values)
    n = len(present)
    if n <= 1:
        return out

    start = 0
    while start < n:
        stop = start
        while stop + 1 < n and present[stop + 1][0] == present[start][0]:
            stop += 1
        shared = (start + stop) / 2 / (n - 1)
        for _, idx in present[start : stop + 1]:
            out[idx] = shared
        start = stop + 1
    return out


def score_lift(percentile: float, strength: float = 1.5) -> float:
    """Map a score percentile to a multiplier on hit probability.

    Calibrated against the competition data rather than assumed: the top 16%
    of the pool by ipTM hit at 28.3% against a 14.0% base rate, so the top of
    the ranking is worth roughly 2x the average design. An exponential in the
    centered percentile reproduces that, giving about 2.1x at the top and
    0.47x at the bottom. ``strength=0`` ignores scores entirely.
    """
    return math.exp(strength * (percentile - 0.5))


@dataclass(frozen=True)
class Selection:
    """The chosen designs plus enough rationale to explain the bill."""

    chosen: tuple[Candidate, ...]
    per_method: Mapping[str, int]
    expected_hits: float
    rationale: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.chosen)

    @property
    def sequences(self) -> dict[str, str]:
        """Sequences keyed by candidate id, ready for an experiment spec."""
        return {c.id: c.sequence for c in self.chosen}


def select_designs(
    candidates: Sequence[Candidate],
    n_slots: int,
    *,
    posteriors: Mapping[str, MethodPosterior] | None = None,
    mode: Mode = "exploit",
    max_method_fraction: float = 0.7,
    min_methods: int = 2,
    score_strength: float = 1.5,
    seed: int | None = None,
) -> Selection:
    """Pick ``n_slots`` designs to send to the lab.

    Each candidate is scored as ``method_rate * score_lift(global score
    percentile)``, then taken greedily best-first subject to a concentration
    cap.

    ``max_method_fraction`` is a safety rail against a small-sample fluke, not
    a diversification mandate. The evidence argues against diversifying: in
    the competition data the binders concentrate heavily in one method, and an
    aggressive cap spends budget moving away from it. The default of 0.7 lets
    a genuinely good method take most of the campaign while still bounding the
    damage if its posterior is wrong. ``min_methods`` reserves a slot for the
    best candidate of each of that many methods, which keeps later rounds
    informative at almost no cost since those are high-scoring designs anyway.

    With ``mode="explore"`` the rate for a method that already has
    observations is a Thompson draw rather than its posterior mean.
    """
    if n_slots <= 0:
        raise ValueError("n_slots must be positive")
    if not candidates:
        return Selection(chosen=(), per_method={}, expected_hits=0.0,
                         rationale=("no candidates supplied",))

    n_slots = min(n_slots, len(candidates))
    rng = random.Random(seed)
    posteriors = posteriors or {}

    by_method: dict[str, list[int]] = {}
    for i, c in enumerate(candidates):
        by_method.setdefault(c.method or UNKNOWN_METHOD, []).append(i)

    # Method rate: posterior mean to exploit, Thompson draw to explore. A
    # method with no observations is left at its prior mean even in explore
    # mode -- sampling it would inject noise into a ranking that is otherwise
    # carrying real signal, without buying information the cap does not
    # already guarantee.
    method_rate: dict[str, float] = {}
    for method in by_method:
        post = _default_posterior(posteriors, method)
        if mode == "explore" and post.tested > 0:
            method_rate[method] = post.sample(rng)
        else:
            method_rate[method] = post.mean

    # Scores are ranked globally, not within method. The competition data says
    # ipTM is largely a *proxy* for method -- its top 60 designs are 23
    # ProteinMPNN entries supplying 13 of the 17 binders found there -- so
    # ranking within method discards exactly the cross-method ordering that
    # makes the metric useful before any history exists.
    pcts = _percentiles([c.score for c in candidates])
    p_hit = [
        method_rate[c.method or UNKNOWN_METHOD] * score_lift(pct, score_strength)
        for c, pct in zip(candidates, pcts)
    ]

    cap = max(1, int(math.floor(n_slots * max_method_fraction)))
    n_methods_available = len(by_method)
    reserve_methods = min(min_methods, n_methods_available)

    # Ties are broken at random rather than by input order, so a caller cannot
    # accidentally profit (or suffer) from how their dataframe happened to be
    # sorted. With metrics this heavily quantized the tie group at the budget
    # cut is large, and its ordering is the single biggest source of variance
    # in what gets bought.
    tiebreak = [rng.random() for _ in candidates]
    order = sorted(range(len(candidates)), key=lambda i: (-p_hit[i], tiebreak[i]))
    chosen: list[int] = []
    per_method: dict[str, int] = {}

    # Reserve one slot for each of the top methods so exploration survives the
    # cap, then fill the rest greedily.
    if reserve_methods and n_slots >= reserve_methods:
        seeded: set[str] = set()
        for i in order:
            m = candidates[i].method or UNKNOWN_METHOD
            if m in seeded:
                continue
            chosen.append(i)
            per_method[m] = per_method.get(m, 0) + 1
            seeded.add(m)
            if len(seeded) == reserve_methods:
                break

    taken = set(chosen)
    for i in order:
        if len(chosen) >= n_slots:
            break
        if i in taken:
            continue
        m = candidates[i].method or UNKNOWN_METHOD
        if per_method.get(m, 0) >= cap:
            continue
        chosen.append(i)
        taken.add(i)
        per_method[m] = per_method.get(m, 0) + 1

    # If the cap left slots unfilled (few methods, many candidates), relax it
    # rather than under-spending a budget the user already approved.
    if len(chosen) < n_slots:
        for i in order:
            if len(chosen) >= n_slots:
                break
            if i in taken:
                continue
            chosen.append(i)
            taken.add(i)
            m = candidates[i].method or UNKNOWN_METHOD
            per_method[m] = per_method.get(m, 0) + 1

    chosen.sort(key=lambda i: -p_hit[i])
    picked = tuple(candidates[i] for i in chosen)
    expected = sum(p_hit[i] for i in chosen)

    rationale = [
        f"mode={mode}, {len(picked)} of {len(candidates)} candidates, "
        f"{len(per_method)} methods, cap {cap}/method",
        f"expected binders {expected:.1f} "
        f"({expected / len(picked) * 100:.0f}% of slots)" if picked else "no slots filled",
    ]
    for m, n in sorted(per_method.items(), key=lambda kv: -kv[1]):
        post = _default_posterior(posteriors, m)
        lo, hi = post.credible_interval()
        rationale.append(
            f"  {m}: {n} slots | posterior {post.mean:.1%} "
            f"[{lo:.1%}-{hi:.1%}] from {post.hits}/{post.tested} prior"
        )

    return Selection(
        chosen=picked,
        per_method=per_method,
        expected_hits=expected,
        rationale=tuple(rationale),
    )
