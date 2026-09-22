"""Synthetic regime-shift benchmark used to validate contextual history."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Iterable

from .context import (
    ContextWeighting,
    ExperimentalContext,
    ExperimentObservation,
    build_contextual_posteriors,
)
from .selection import Candidate, MethodStats, build_posteriors, select_designs

METHODS = ("method_a", "method_b")
PRE_RATES = {"method_a": 0.45, "method_b": 0.10}
POST_RATES = {"method_a": 0.10, "method_b": 0.45}
HISTORY_PER_METHOD = 120
HISTORY_AGE_DAYS = 180
ROUND_SLOTS = 20
N_ROUNDS = 6
ROUND_DAYS = 7
HALF_LIFE_DAYS = 90.0
MISMATCH_PENALTY = 0.10
BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)

OLD_CONTEXT = ExperimentalContext(
    target="synthetic-target",
    assay="binding",
    protocol_version="protocol-v1",
    model_version="model-v1",
    reagent_lot="lot-1",
    instrument="instrument-1",
)
CURRENT_CONTEXT = ExperimentalContext(
    target="synthetic-target",
    assay="binding",
    protocol_version="protocol-v2",
    model_version="model-v1",
    reagent_lot="lot-1",
    instrument="instrument-1",
)


@dataclass
class TrialResult:
    binders: int
    tested: int
    stale_tests: int
    recovery_round: int


def _uniform(key: str) -> float:
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _hit(seed: int, key: str, rate: float) -> bool:
    return _uniform(f"{seed}:{key}") < rate


def _initial_history(seed: int) -> list[ExperimentObservation]:
    observed_at = BASE_TIME - timedelta(days=HISTORY_AGE_DAYS)
    observations: list[ExperimentObservation] = []
    for method in METHODS:
        for i in range(HISTORY_PER_METHOD):
            observations.append(
                ExperimentObservation(
                    method=method,
                    hit=_hit(seed, f"history:{method}:{i}", PRE_RATES[method]),
                    observed_at=observed_at,
                    context=OLD_CONTEXT,
                )
            )
    return observations


def _method_stats(observations: Iterable[ExperimentObservation]) -> list[MethodStats]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for observation in observations:
        if not observation.is_biological_evidence:
            continue
        rec = counts[observation.method]
        rec[0] += 1
        rec[1] += int(bool(observation.hit))
    return [
        MethodStats(method=method, tested=tested, hits=hits)
        for method, (tested, hits) in counts.items()
    ]


def _posteriors(planner: str, observations: list[ExperimentObservation], now: datetime):
    if planner == "context-free":
        return build_posteriors(_method_stats(observations))

    if planner == "recency-only":
        normalized = [
            ExperimentObservation(
                method=o.method,
                hit=o.hit,
                observed_at=o.observed_at,
                context=CURRENT_CONTEXT,
                kind=o.kind,
                metadata=o.metadata,
            )
            for o in observations
        ]
        return build_contextual_posteriors(
            normalized,
            current_context=CURRENT_CONTEXT,
            now=now,
            weighting=ContextWeighting(
                half_life_days=HALF_LIFE_DAYS,
                mismatch_penalty=1.0,
                unknown_penalty=1.0,
            ),
        )

    if planner == "context+recency":
        return build_contextual_posteriors(
            observations,
            current_context=CURRENT_CONTEXT,
            now=now,
            weighting=ContextWeighting(
                half_life_days=HALF_LIFE_DAYS,
                mismatch_penalty=MISMATCH_PENALTY,
                unknown_penalty=1.0,
            ),
        )

    raise ValueError(f"unknown planner: {planner}")


def _candidates(round_index: int) -> list[Candidate]:
    return [
        Candidate(
            id=f"r{round_index}:{method}:{i}",
            sequence="X",
            method=method,
            score=None,
        )
        for method in METHODS
        for i in range(ROUND_SLOTS * 2)
    ]


def run_trial(seed: int, planner: str) -> TrialResult:
    observations = _initial_history(seed)
    total_hits = 0
    total_tested = 0
    stale_tests = 0
    recovery_round = N_ROUNDS + 1

    for round_index in range(1, N_ROUNDS + 1):
        now = BASE_TIME + timedelta(days=(round_index - 1) * ROUND_DAYS)
        selection = select_designs(
            _candidates(round_index),
            ROUND_SLOTS,
            posteriors=_posteriors(planner, observations, now),
            mode="exploit",
            max_method_fraction=1.0,
            min_methods=1,
            score_strength=0.0,
            seed=seed * 100 + round_index,
        )

        selected_b = sum(c.method == "method_b" for c in selection.chosen)
        if selected_b >= ROUND_SLOTS // 2 and recovery_round == N_ROUNDS + 1:
            recovery_round = round_index

        for candidate in selection.chosen:
            hit = _hit(seed, f"post:{candidate.id}", POST_RATES[candidate.method])
            total_hits += int(hit)
            total_tested += 1
            stale_tests += int(candidate.method == "method_a")
            observations.append(
                ExperimentObservation(
                    method=candidate.method,
                    hit=hit,
                    observed_at=now,
                    context=CURRENT_CONTEXT,
                )
            )

    return TrialResult(
        binders=total_hits,
        tested=total_tested,
        stale_tests=stale_tests,
        recovery_round=recovery_round,
    )


def run_benchmark(trials: int = 400) -> dict[str, dict[str, float]]:
    planners = ("context-free", "recency-only", "context+recency")
    summary: dict[str, dict[str, float]] = {}
    for planner in planners:
        rows = [run_trial(seed, planner) for seed in range(trials)]
        summary[planner] = {
            "binders": mean(r.binders for r in rows),
            "hit_rate": mean(r.binders / r.tested for r in rows),
            "stale_tests": mean(r.stale_tests for r in rows),
            "recovery_round": mean(r.recovery_round for r in rows),
        }
    return summary
