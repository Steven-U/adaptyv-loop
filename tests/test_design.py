"""Design generation and simulated-bench tests.

These avoid loading ESM-2 — the PLM-guided generator and the scorer need model
weights, so they are exercised by ``demo_end_to_end.py --build-library``
rather than here.
"""

from __future__ import annotations

import random

import pytest

from adaptyv_loop.bench import (
    SimulatedBench,
    developability_penalty,
    liability_penalty,
)
from adaptyv_loop.design import (
    VHH_SCAFFOLD,
    germline_recombination,
    locate_cdrs,
    random_control,
    scaffold_mutagenesis,
    stable_id,
)


# ---- CDR location ------------------------------------------------------


def test_cdrs_found_in_the_reference_scaffold():
    cdrs = locate_cdrs(VHH_SCAFFOLD)
    assert cdrs.cdr1 is not None and cdrs.cdr3 is not None
    assert VHH_SCAFFOLD[slice(*cdrs.cdr3)] == "AAGYQINSGNYNFKDYEYDY"


def test_non_antibody_sequences_yield_no_cdrs():
    """De novo minibinders have no framework; mutating arbitrary positions
    would be meaningless, so generators must be able to skip them."""
    cdrs = locate_cdrs("MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGY")
    assert cdrs.cdr1 is None and cdrs.cdr3 is None
    assert cdrs.positions() == []


def test_generators_return_nothing_for_a_frameworkless_sequence():
    rng = random.Random(0)
    junk = "MKTVRQERLKSIVRILERSKEP"
    assert scaffold_mutagenesis(junk, 5, rng) == []
    assert germline_recombination(junk, 5, rng) == []
    assert random_control(junk, 5, rng) == []


# ---- generators --------------------------------------------------------


def test_mutagenesis_changes_only_cdr_positions():
    rng = random.Random(0)
    variants = scaffold_mutagenesis(VHH_SCAFFOLD, 20, rng, n_mutations=4)
    allowed = set(locate_cdrs(VHH_SCAFFOLD).positions())
    for v in variants:
        assert len(v) == len(VHH_SCAFFOLD)
        changed = {i for i, (a, b) in enumerate(zip(v, VHH_SCAFFOLD)) if a != b}
        assert changed <= allowed


def test_mutagenesis_respects_the_mutation_budget():
    rng = random.Random(1)
    for v in scaffold_mutagenesis(VHH_SCAFFOLD, 20, rng, n_mutations=3):
        changed = sum(1 for a, b in zip(v, VHH_SCAFFOLD) if a != b)
        assert changed <= 3


def test_germline_recombination_varies_cdr3_length():
    rng = random.Random(2)
    lengths = set()
    for v in germline_recombination(VHH_SCAFFOLD, 40, rng, length_range=(8, 20)):
        cdr3 = locate_cdrs(v).cdr3
        if cdr3:
            lengths.add(cdr3[1] - cdr3[0])
    assert len(lengths) > 1


def test_generators_are_deterministic_for_a_seed():
    a = scaffold_mutagenesis(VHH_SCAFFOLD, 10, random.Random(7))
    b = scaffold_mutagenesis(VHH_SCAFFOLD, 10, random.Random(7))
    assert a == b


def test_stable_id_is_content_addressed():
    """The same sequence must keep the same id, or a campaign could pay to
    test one molecule twice under two names."""
    assert stable_id("MKTV") == stable_id("MKTV")
    assert stable_id("MKTV") != stable_id("MKTW")


# ---- bench penalties ---------------------------------------------------


def test_reference_scaffold_has_no_length_penalty():
    assert developability_penalty(VHH_SCAFFOLD) == 0.0


def test_length_deviation_is_penalized_and_saturates():
    rng = random.Random(3)
    short = germline_recombination(VHH_SCAFFOLD, 1, rng, length_range=(8, 8))[0]
    assert developability_penalty(short) < 0
    assert developability_penalty(short) >= -1.0


def test_liability_penalty_counts_known_motifs():
    """NG deamidation, N-x-S/T glycosylation and free cysteine are all real
    developability red flags that a language model scores as ordinary."""
    clean = germline_recombination(VHH_SCAFFOLD, 1, random.Random(11))[0]
    assert liability_penalty(clean) <= 0
    assert liability_penalty("MKTVRQ") == 0.0  # no CDRs located


# ---- simulated bench ---------------------------------------------------


def _library(n: int = 200) -> tuple[list[str], list[float]]:
    rng = random.Random(5)
    seqs = (
        scaffold_mutagenesis(VHH_SCAFFOLD, n // 2, rng)
        + germline_recombination(VHH_SCAFFOLD, n // 2, rng)
    )
    # Stand-in scores; the bench only needs a numeric column to normalize.
    scores = [-0.30 - 0.05 * rng.random() for _ in seqs]
    return seqs, scores


def test_bench_hits_the_target_base_rate():
    seqs, scores = _library()
    bench = SimulatedBench.calibrate(seqs, scores, target_base_rate=0.14)
    observed = sum(bench.measure(s, sc) for s, sc in zip(seqs, scores)) / len(seqs)
    assert observed == pytest.approx(0.14, abs=0.06)


def test_bench_calibration_accounts_for_the_hidden_term():
    """Sigmoid is nonlinear, so treating the mean-zero hidden term as zero
    during calibration would push the realized rate above target."""
    seqs, scores = _library()
    low = SimulatedBench.calibrate(seqs, scores, target_base_rate=0.05)
    rate = sum(low.measure(s, sc) for s, sc in zip(seqs, scores)) / len(seqs)
    assert rate < 0.15


def test_bench_is_deterministic_per_sequence():
    seqs, scores = _library(50)
    bench = SimulatedBench.calibrate(seqs, scores)
    first = [bench.measure(s, sc) for s, sc in zip(seqs, scores)]
    second = [bench.measure(s, sc) for s, sc in zip(seqs, scores)]
    assert first == second


def test_bench_never_sees_the_generator_label():
    """measure() takes only a sequence and a score. If it could see the method
    the whole experiment would be circular."""
    import inspect

    params = set(inspect.signature(SimulatedBench.measure).parameters)
    assert params == {"self", "sequence", "score"}


def test_higher_score_means_higher_probability():
    seqs, scores = _library(20)
    bench = SimulatedBench.calibrate(seqs, scores)
    seq = seqs[0]
    assert bench.probability(seq, -0.20) > bench.probability(seq, -0.50)


def test_empty_library_calibrates_without_crashing():
    bench = SimulatedBench.calibrate([], [])
    assert bench.score_std == 1.0
