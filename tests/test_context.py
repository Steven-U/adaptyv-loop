from datetime import datetime, timedelta, timezone

import pytest

from adaptyv_loop.context import (
    ContextWeighting,
    ExperimentalContext,
    ExperimentObservation,
    ObservationKind,
    build_contextual_posteriors,
)
from adaptyv_loop.selection import (
    Candidate,
    MethodStats,
    UNKNOWN_METHOD,
    build_posteriors,
    select_designs,
)


NOW = datetime(2026, 9, 22, tzinfo=timezone.utc)
CTX = ExperimentalContext(
    target="EGFR",
    assay="binding",
    protocol_version="p2",
    model_version="m2",
    reagent_lot="lot-b",
    instrument="octet-2",
)


def obs(
    method: str,
    hit: bool | None,
    *,
    days_ago: int = 0,
    context: ExperimentalContext = CTX,
    kind: ObservationKind = ObservationKind.BIOLOGICAL,
) -> ExperimentObservation:
    return ExperimentObservation(
        method=method,
        hit=hit,
        observed_at=NOW - timedelta(days=days_ago),
        context=context,
        kind=kind,
    )


def test_qc_and_technical_failures_do_not_become_biological_negatives():
    posts = build_contextual_posteriors(
        [
            obs("method-a", True),
            obs("method-a", None, kind=ObservationKind.QC_FAILURE),
            obs("method-a", False, kind=ObservationKind.TECHNICAL_FAILURE),
        ],
        current_context=CTX,
        now=NOW,
    )
    post = posts["method-a"]
    assert post.tested == 1
    assert post.hits == 1
    assert post.effective_tested == pytest.approx(1.0)
    assert post.effective_hits == pytest.approx(1.0)


def test_target_mismatch_is_ignored_by_default():
    other = ExperimentalContext(target="HER2", assay="binding")
    posts = build_contextual_posteriors(
        [obs("method-a", True, context=other)],
        current_context=CTX,
        now=NOW,
    )
    assert "method-a" not in posts


def test_recent_matching_evidence_counts_more_than_stale_mismatched_context():
    old_context = ExperimentalContext(
        target="EGFR",
        assay="binding",
        protocol_version="p1",
        model_version="m1",
        reagent_lot="lot-a",
        instrument="octet-1",
    )
    posts = build_contextual_posteriors(
        [
            obs("method-a", False, days_ago=180, context=old_context),
            obs("method-a", True, days_ago=0, context=CTX),
        ],
        current_context=CTX,
        now=NOW,
        weighting=ContextWeighting(half_life_days=90),
    )
    post = posts["method-a"]
    assert post.effective_hits > (post.effective_tested - post.effective_hits)
    assert post.mean > 0.14


def test_unknown_method_history_is_never_learned():
    posts = build_contextual_posteriors(
        [obs(UNKNOWN_METHOD, True), obs(UNKNOWN_METHOD, True)],
        current_context=CTX,
        now=NOW,
    )
    assert posts[UNKNOWN_METHOD].tested == 0
    assert posts[UNKNOWN_METHOD].hits == 0
    assert posts[UNKNOWN_METHOD].mean == pytest.approx(0.14)


def test_contextual_posteriors_can_drive_existing_selector():
    posts = build_contextual_posteriors(
        [
            obs("good", True),
            obs("good", True),
            obs("bad", False),
            obs("bad", False),
        ],
        current_context=CTX,
        now=NOW,
    )
    candidates = [
        Candidate(id="g1", sequence="AAA", method="good", score=0.5),
        Candidate(id="g2", sequence="AAB", method="good", score=0.5),
        Candidate(id="b1", sequence="BBB", method="bad", score=0.5),
        Candidate(id="b2", sequence="BBC", method="bad", score=0.5),
    ]
    selection = select_designs(
        candidates,
        2,
        posteriors=posts,
        mode="exploit",
        min_methods=1,
        max_method_fraction=1.0,
        seed=0,
    )
    assert {c.method for c in selection.chosen} == {"good"}


def test_legacy_posterior_does_not_learn_unknown_bucket():
    posts = build_posteriors(
        [
            MethodStats(method=UNKNOWN_METHOD, tested=13, hits=5),
            MethodStats(method="known", tested=10, hits=2),
        ]
    )
    assert posts[UNKNOWN_METHOD].tested == 0
    assert posts[UNKNOWN_METHOD].hits == 0
