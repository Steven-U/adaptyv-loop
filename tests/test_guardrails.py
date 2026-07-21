"""Guardrail tests.

These are the tests that matter most in this package: everything here is a
control standing between an automated loop and a real invoice.
"""

from __future__ import annotations

import pytest

from adaptyv_loop.client import CostEstimate
from adaptyv_loop.errors import GuardrailViolation
from adaptyv_loop.guardrails import DecisionLog, SpendGuard, SpendPolicy

SPEC = {"experiment_type": "screening", "sequences": {"a": "MKT", "b": "MKV"}}


def estimate(usd: float | None, *, complete: bool = True) -> CostEstimate:
    return CostEstimate(
        total_cents=int(usd * 100) if complete and usd is not None else None,
        pricing_version="v1_test",
    )


@pytest.fixture()
def guard(tmp_path):
    return SpendGuard(
        SpendPolicy(max_experiment_usd=1_000, max_campaign_usd=2_500),
        DecisionLog(tmp_path / "decisions.jsonl"),
        campaign_id="test",
    )


def test_dry_run_then_approve_commits_budget(guard):
    guard.dry_run(SPEC, estimate(400))
    decision = guard.approve(SPEC)
    assert decision.approved
    assert guard.spent_usd == pytest.approx(400)
    assert guard.remaining_usd == pytest.approx(2_100)


def test_approve_without_dry_run_is_refused(guard):
    with pytest.raises(GuardrailViolation, match="no passing dry run"):
        guard.approve(SPEC)
    assert guard.spent_usd == 0


def test_edited_spec_invalidates_the_dry_run(guard):
    """A dry run authorizes one exact spec, not "roughly that experiment"."""
    guard.dry_run(SPEC, estimate(400))
    edited = {**SPEC, "sequences": {**SPEC["sequences"], "c": "MKW"}}
    with pytest.raises(GuardrailViolation, match="no passing dry run"):
        guard.approve(edited)
    assert guard.spent_usd == 0


def test_per_experiment_ceiling(guard):
    with pytest.raises(GuardrailViolation, match="per-experiment ceiling"):
        guard.dry_run(SPEC, estimate(1_500))


def test_campaign_budget_is_cumulative(guard):
    for _ in range(2):
        guard.dry_run(SPEC, estimate(1_000))
        guard.approve(SPEC)
    assert guard.spent_usd == pytest.approx(2_000)

    with pytest.raises(GuardrailViolation, match="remaining campaign budget"):
        guard.dry_run(SPEC, estimate(1_000))


def test_incomplete_estimate_is_refused(guard):
    """Foundry returns an incomplete estimate for unpriced targets.

    Unknown cost must not be treated as zero cost.
    """
    with pytest.raises(GuardrailViolation, match="incomplete cost estimate"):
        guard.dry_run(SPEC, estimate(None, complete=False))


def test_auto_accept_requires_explicit_policy(guard):
    guard.dry_run(SPEC, estimate(100))
    with pytest.raises(GuardrailViolation, match="allow_auto_accept"):
        guard.approve(SPEC, auto_accept_quote=True)
    assert guard.spent_usd == 0


def test_auto_accept_allowed_when_armed(tmp_path):
    guard = SpendGuard(
        SpendPolicy(
            max_experiment_usd=1_000, max_campaign_usd=1_000, allow_auto_accept=True
        ),
        DecisionLog(tmp_path / "d.jsonl"),
    )
    guard.dry_run(SPEC, estimate(100))
    assert guard.approve(SPEC, auto_accept_quote=True).approved


def test_budget_survives_process_restart(tmp_path):
    """A crashed campaign must not resume with a fresh budget."""
    policy = SpendPolicy(max_experiment_usd=1_000, max_campaign_usd=1_200)
    path = tmp_path / "d.jsonl"

    first = SpendGuard(policy, DecisionLog(path))
    first.dry_run(SPEC, estimate(1_000))
    first.approve(SPEC)

    resumed = SpendGuard(policy, DecisionLog(path))
    assert resumed.spent_usd == pytest.approx(1_000)
    with pytest.raises(GuardrailViolation):
        resumed.dry_run(SPEC, estimate(1_000))


def test_refusals_are_logged_not_just_raised(guard):
    with pytest.raises(GuardrailViolation):
        guard.dry_run(SPEC, estimate(9_999))
    entries = guard.log.replay()
    assert len(entries) == 1
    assert entries[0].approved is False
    assert "ceiling" in entries[0].reason


def test_approval_reuse_is_refused(guard):
    """One dry run authorizes one submission, so a retry loop cannot double-bill."""
    guard.dry_run(SPEC, estimate(400))
    guard.approve(SPEC)
    with pytest.raises(GuardrailViolation, match="no passing dry run"):
        guard.approve(SPEC)
    assert guard.spent_usd == pytest.approx(400)


def test_policy_rejects_incoherent_limits():
    with pytest.raises(ValueError):
        SpendPolicy(max_experiment_usd=5_000, max_campaign_usd=1_000)
    with pytest.raises(ValueError):
        SpendPolicy(max_experiment_usd=0, max_campaign_usd=1_000)
