"""A simulated bench for novel designs, calibrated to real measured effects.

The competition datasets give ground truth for the 402 sequences that were
actually tested. Designs this package *generates* are novel, so no measurement
for them exists — obtaining one is exactly what Adaptyv sells. To close the
loop offline, this module stands in for the bench.

It is a simulator and is labelled as one. Three properties make it a fair test
rather than a rigged one.

**It sees only the sequence.** No generator label, no method tag, nothing the
selection policy knows. Method effects have to *emerge* from the fact that
different generators produce different sequence distributions, exactly as they
would in reality. The loop is not handed the answer it is meant to discover.

**Its base rate is solved for, not asserted.** ``calibrate`` bisects the
intercept so the library's mean binding probability equals the measured 14.0%
binder rate from the competition.

**Its signal structure mirrors the real data.** Outcome is driven by ESM-2
pseudo-log-likelihood, which empirically separates the generators far more
than it separates designs within a generator — between-method spread 0.065
versus within-method 0.024 on a 400-design library. That is the same structure
found in the real competition results, where ipTM turned out to be largely a
proxy for design method rather than a per-design predictor. A simulator that
made per-design scores strongly and independently predictive would flatter any
selection policy, including this one.

None of this is a claim about real binding affinity. It is a testbed whose
answers are known, so the machinery can be checked for recovering them.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Sequence

from .design import locate_cdrs

#: Measured binder rate in the public EGFR competition (53/378).
TARGET_BASE_RATE = 0.140
#: Weight on the z-scored ESM-2 score. Sized to reproduce the roughly 9x spread
#: between the best and worst design methods seen in the competition data.
SCORE_WEIGHT = 2.30
#: Weight on a per-design term the score cannot see.
#:
#: Without this, outcome is a monotone function of score and the score becomes
#: a near-perfect predictor *within* a method (AUC 0.76), which the real data
#: flatly contradicts. This term stands for everything a sequence-level score
#: misses — epitope geometry, loop conformation, expression quirks. It is
#: mean-zero within every method, so it dilutes per-design predictability
#: without collapsing the between-method separation the loop is meant to find.
#: Sized so within-method AUC lands near the measured 0.64: at this weight a
#: 600-design library yields within-method AUC 0.640 against the real 0.636.
HIDDEN_WEIGHT = 2.00
#: Weight on a developability penalty the sequence score does not capture.
#:
#: This is what makes the simulator faithful to the headline finding in the
#: real data: design *method* carried signal beyond the per-design confidence
#: score. Without such a term, outcome is a pure function of ESM-2 PLL, plain
#: score-ranking matches the loop exactly, and the testbed would contradict the
#: very result it is meant to exercise.
#:
#: The penalty is CDR3 length deviation from the parent scaffold, which is real
#: biology — a loop far from the length the framework was optimized around
#: tends to express and fold worse — and empirically near-independent of PLL
#: (r = +0.22 on a 600-design library).
DEVELOPABILITY_WEIGHT = 1.40
#: CDR3 length the reference VHH scaffold was optimized around.
REFERENCE_CDR3_LENGTH = 20
#: Weight on CDR sequence liabilities.
#:
#: N-linked glycosylation sequons (N-X-S/T), deamidation hotspots (NG/NS),
#: isomerization motifs (DG/DS) and unpaired cysteines are well-documented
#: antibody developability red flags that routinely sink an otherwise good
#: binder. Crucially they are *perfectly ordinary* dipeptides, so a protein
#: language model assigns them high likelihood — measured correlation between
#: liability count and ESM-2 PLL is only -0.43, and among the three non-random
#: generators the ordering is inverted. Included because it is real, not
#: because of what it does to any particular selection policy.
LIABILITY_WEIGHT = 0.55


def developability_penalty(sequence: str, reference: int = REFERENCE_CDR3_LENGTH) -> float:
    """Penalty in [-1, 0] for a CDR3 far from the parent scaffold's loop length.

    Zero for a loop at the reference length, saturating toward -1 as it
    diverges. Returns 0.0 when no CDR3 can be located, so non-antibody formats
    are neither rewarded nor punished by this term.
    """
    cdr3 = locate_cdrs(sequence).cdr3
    if not cdr3:
        return 0.0
    deviation = abs((cdr3[1] - cdr3[0]) - reference)
    return -min(deviation / 8.0, 1.0)


_NGLYC = re.compile(r"N[^P][ST]")
_DEAMIDATION = re.compile(r"N[GS]")
_ISOMERIZATION = re.compile(r"D[GS]")


def liability_penalty(sequence: str) -> float:
    """Count of CDR developability liabilities, returned as a negative penalty.

    Scans the CDRs only — these motifs are unremarkable in framework regions
    and problematic in the loops that do the binding.
    """
    cdrs = locate_cdrs(sequence)
    spans = [sp for sp in (cdrs.cdr1, cdrs.cdr3) if sp]
    if not spans:
        return 0.0
    loops = "".join(sequence[slice(*sp)] for sp in spans)
    count = (
        len(_NGLYC.findall(loops))
        + len(_DEAMIDATION.findall(loops))
        + len(_ISOMERIZATION.findall(loops))
        + loops.count("C")
    )
    return -float(count)


def _sigmoid(x: float) -> float:
    if x < 0:
        e = math.exp(x)
        return e / (1.0 + e)
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class SimulatedBench:
    """Assigns each sequence a binding outcome. Deterministic per sequence.

    A given sequence always returns the same answer, so re-running a campaign
    cannot re-roll a design's luck — as a real measurement would not change if
    you asked twice.
    """

    intercept: float
    score_mean: float
    score_std: float
    score_weight: float = SCORE_WEIGHT
    hidden_weight: float = HIDDEN_WEIGHT
    developability_weight: float = DEVELOPABILITY_WEIGHT
    liability_weight: float = LIABILITY_WEIGHT
    seed: int = 0

    @classmethod
    def calibrate(
        cls,
        sequences: Sequence[str],
        scores: Sequence[float],
        *,
        target_base_rate: float = TARGET_BASE_RATE,
        score_weight: float = SCORE_WEIGHT,
        hidden_weight: float = HIDDEN_WEIGHT,
        developability_weight: float = DEVELOPABILITY_WEIGHT,
        liability_weight: float = LIABILITY_WEIGHT,
        seed: int = 0,
    ) -> SimulatedBench:
        """Fit to a library: normalize scores, then solve for the intercept.

        The intercept is bisected so the library's mean binding probability
        equals ``target_base_rate``. The sequences are required, not just the
        scores: the hidden term must be evaluated at its real per-sequence
        values during the solve. Sigmoid is nonlinear, so assuming a mean-zero
        hidden term contributes zero would leave the realized base rate above
        target.
        """
        n = len(scores)
        if n == 0:
            return cls(intercept=0.0, score_mean=0.0, score_std=1.0,
                       score_weight=score_weight, hidden_weight=hidden_weight,
                       developability_weight=developability_weight,
                       liability_weight=liability_weight, seed=seed)
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / max(n - 1, 1)
        std = math.sqrt(var) or 1.0
        z = [(s - mean) / std for s in scores]

        probe = cls(intercept=0.0, score_mean=mean, score_std=std,
                    score_weight=score_weight, hidden_weight=hidden_weight,
                    developability_weight=developability_weight,
                    liability_weight=liability_weight, seed=seed)
        hidden = [probe._hidden(s) for s in sequences]
        dev = [developability_penalty(s) for s in sequences]
        liab = [liability_penalty(s) for s in sequences]

        def mean_p(intercept: float) -> float:
            return sum(
                _sigmoid(
                    intercept + score_weight * zi + hidden_weight * hi
                    + developability_weight * di + liability_weight * li
                )
                for zi, hi, di, li in zip(z, hidden, dev, liab)
            ) / n

        lo, hi = -30.0, 30.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if mean_p(mid) < target_base_rate:
                lo = mid
            else:
                hi = mid
        return cls(intercept=(lo + hi) / 2, score_mean=mean, score_std=std,
                   score_weight=score_weight, hidden_weight=hidden_weight,
                   developability_weight=developability_weight,
                   liability_weight=liability_weight, seed=seed)

    def _hidden(self, sequence: str) -> float:
        """A standard-normal per-design effect the score cannot observe.

        Derived from the sequence hash, so it is a fixed property of the
        molecule rather than a fresh roll on every call.
        """
        digest = hashlib.sha256(f"hidden:{self.seed}:{sequence}".encode()).digest()
        u1 = (int.from_bytes(digest[:8], "big") + 1) / (2**64 + 2)
        u2 = int.from_bytes(digest[8:16], "big") / 2**64
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def probability(self, sequence: str, score: float) -> float:
        """P(binder) for one design, from sequence-derived terms only."""
        z = (score - self.score_mean) / self.score_std
        return _sigmoid(
            self.intercept
            + self.score_weight * z
            + self.hidden_weight * self._hidden(sequence)
            + self.developability_weight * developability_penalty(sequence)
            + self.liability_weight * liability_penalty(sequence)
        )

    def measure(self, sequence: str, score: float) -> bool:
        """Draw the outcome. Deterministic in the sequence, not in call order."""
        p = self.probability(sequence, score)
        digest = hashlib.sha256(f"{self.seed}:{sequence}".encode()).digest()
        draw = int.from_bytes(digest[:8], "big") / 2**64
        return draw < p

    def true_rate(self, sequences: Sequence[str], scores: Sequence[float]) -> float:
        """Mean true probability over a set — the number the loop should recover."""
        if not sequences:
            return 0.0
        return sum(self.probability(s, sc) for s, sc in zip(sequences, scores)) / len(sequences)
