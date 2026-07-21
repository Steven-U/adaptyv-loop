"""Selection policy tests."""

from __future__ import annotations

import pytest

from adaptyv_loop.selection import (
    Candidate,
    MethodStats,
    build_posteriors,
    _percentiles,
    score_lift,
    select_designs,
)


def make(n: int, method: str, score_base: float = 0.5) -> list[Candidate]:
    return [
        Candidate(id=f"{method}-{i}", sequence="MKT", method=method, score=score_base + i * 0.01)
        for i in range(n)
    ]


# ---- posteriors --------------------------------------------------------


def test_shrinkage_tames_a_small_sample():
    """3-for-3 must not read as a 100% method."""
    posts = build_posteriors([MethodStats("lucky", tested=3, hits=3)])
    assert posts["lucky"].mean < 0.6


def test_more_evidence_moves_the_posterior_further():
    thin = build_posteriors([MethodStats("m", tested=3, hits=3)])["m"]
    thick = build_posteriors([MethodStats("m", tested=60, hits=60)])["m"]
    assert thick.mean > thin.mean
    assert thick.stdev < thin.stdev


def test_zero_hit_pilot_does_not_collapse_the_prior():
    """A pilot that finds nothing is common at a ~14% base rate.

    Taking its raw 0.0 as the prior mean would flatten every method to zero
    and make the next round's ranking arbitrary.
    """
    posts = build_posteriors(
        [MethodStats("a", tested=10, hits=0), MethodStats("b", tested=10, hits=0)]
    )
    assert all(p.mean > 0.01 for p in posts.values())


def test_unseen_method_falls_back_to_the_prior():
    posts = build_posteriors([MethodStats("known", tested=20, hits=10)])
    sel = select_designs(make(4, "brand-new"), 2, posteriors=posts)
    assert len(sel) == 2


# ---- percentiles and ties ---------------------------------------------


def test_ties_share_a_percentile():
    """Heavily quantized metrics tie constantly; row order is not information."""
    pcts = _percentiles([0.9, 0.9, 0.9, 0.1])
    assert pcts[0] == pcts[1] == pcts[2]
    assert pcts[3] < pcts[0]


def test_percentiles_handle_missing_scores():
    pcts = _percentiles([1.0, None, 0.0])
    assert pcts[1] == 0.5
    assert pcts[0] > pcts[2]


def test_score_lift_is_monotone_and_calibrated():
    assert score_lift(1.0) > score_lift(0.5) > score_lift(0.0)
    assert score_lift(0.5) == pytest.approx(1.0)
    # Top of the ranking is worth ~2x the average design, matching the
    # measured 28.3% vs 14.0% hit rate in the competition data.
    assert 1.8 < score_lift(1.0) < 2.5
    assert score_lift(1.0, strength=0.0) == pytest.approx(1.0)


# ---- allocation --------------------------------------------------------


def test_budget_is_filled_exactly():
    sel = select_designs(make(10, "a") + make(10, "b"), 12)
    assert len(sel) == 12
    assert sum(sel.per_method.values()) == 12


def test_cannot_oversubscribe_a_small_pool():
    sel = select_designs(make(3, "a"), 10)
    assert len(sel) == 3


def test_budget_is_filled_even_when_the_cap_binds():
    """A cap must never leave an approved budget under-spent."""
    sel = select_designs(make(20, "only"), 10, max_method_fraction=0.3)
    assert len(sel) == 10


def test_good_method_wins_the_allocation():
    posts = build_posteriors(
        [MethodStats("good", tested=40, hits=20), MethodStats("bad", tested=40, hits=1)]
    )
    sel = select_designs(make(30, "good") + make(30, "bad"), 20, posteriors=posts)
    assert sel.per_method["good"] > sel.per_method.get("bad", 0)


def test_concentration_cap_bounds_a_single_method():
    posts = build_posteriors(
        [MethodStats("good", tested=40, hits=39), MethodStats("bad", tested=40, hits=1)]
    )
    sel = select_designs(
        make(30, "good") + make(30, "bad"), 20, posteriors=posts, max_method_fraction=0.5
    )
    assert sel.per_method["good"] <= 10


def test_min_methods_keeps_later_rounds_informative():
    posts = build_posteriors(
        [MethodStats("good", tested=40, hits=39), MethodStats("bad", tested=40, hits=1)]
    )
    sel = select_designs(
        make(30, "good") + make(30, "bad"), 10, posteriors=posts, min_methods=2
    )
    assert len(sel.per_method) >= 2


def test_higher_scores_preferred_within_a_method():
    cands = make(20, "a")
    sel = select_designs(cands, 5, score_strength=1.5, seed=0)
    chosen = {c.id for c in sel.chosen}
    top = {c.id for c in sorted(cands, key=lambda c: -c.score)[:5]}
    assert chosen == top


def test_selection_is_deterministic_for_a_seed():
    cands = make(15, "a") + make(15, "b")
    a = select_designs(cands, 10, seed=7)
    b = select_designs(cands, 10, seed=7)
    assert [c.id for c in a.chosen] == [c.id for c in b.chosen]


def test_ties_are_broken_randomly_not_by_row_order():
    """Otherwise a caller silently profits from how their dataframe was sorted."""
    cands = [Candidate(id=str(i), sequence="M", method="a", score=0.5) for i in range(40)]
    picks = {tuple(c.id for c in select_designs(cands, 5, seed=s).chosen) for s in range(12)}
    assert len(picks) > 1


def test_sequences_are_ready_for_an_experiment_spec():
    sel = select_designs(make(5, "a"), 3)
    assert sel.sequences == {c.id: c.sequence for c in sel.chosen}


def test_rejects_a_nonpositive_budget():
    with pytest.raises(ValueError):
        select_designs(make(5, "a"), 0)


def test_empty_pool_returns_an_empty_selection():
    sel = select_designs([], 5)
    assert len(sel) == 0 and sel.expected_hits == 0.0
