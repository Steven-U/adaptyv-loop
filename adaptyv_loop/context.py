"""Context-aware experimental evidence for adaptive campaign planning.

The original adaptyv-loop learns one hit rate per declared design method. That
is intentionally simple and remains the baseline. This module adds a richer,
still-auditable layer: evidence is weighted by experimental context and recency,
and technical/QC failures are kept out of the biological hit-rate posterior.

Nothing here claims that the chosen context weights are biologically optimal.
They are explicit policy knobs to benchmark against the context-free baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Mapping

from .selection import MethodPosterior, REFERENCE_BASE_RATE, UNKNOWN_METHOD


class ObservationKind(str, Enum):
    """Why an experiment produced the observation we received."""

    BIOLOGICAL = "biological"
    QC_FAILURE = "qc_failure"
    TECHNICAL_FAILURE = "technical_failure"


@dataclass(frozen=True)
class ExperimentalContext:
    """State that can change whether old experimental evidence is relevant."""

    target: str
    assay: str | None = None
    protocol_version: str | None = None
    model_version: str | None = None
    reagent_lot: str | None = None
    instrument: str | None = None


@dataclass(frozen=True)
class ExperimentObservation:
    """One historical result plus the state under which it was produced."""

    method: str
    hit: bool | None
    observed_at: datetime
    context: ExperimentalContext
    kind: ObservationKind = ObservationKind.BIOLOGICAL
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def is_biological_evidence(self) -> bool:
        """Whether this observation should update a binder-rate posterior."""
        return self.kind is ObservationKind.BIOLOGICAL and self.hit is not None


@dataclass(frozen=True)
class ContextWeighting:
    """Transparent policy for deciding how much historical evidence counts.

    Target mismatch is strict by default because the existing Adaptyv result
    has not demonstrated cross-target transfer. Other mismatches are discounted
    rather than deleted so they can still contribute weak prior evidence.
    """

    half_life_days: float = 90.0
    mismatch_penalty: float = 0.25
    unknown_penalty: float = 0.60
    strict_target: bool = True

    def __post_init__(self) -> None:
        if self.half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        for name, value in (
            ("mismatch_penalty", self.mismatch_penalty),
            ("unknown_penalty", self.unknown_penalty),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass
class ContextualMethodPosterior(MethodPosterior):
    """A method posterior with the weighted evidence exposed for inspection."""

    effective_tested: float = 0.0
    effective_hits: float = 0.0


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def recency_weight(
    observed_at: datetime,
    *,
    now: datetime,
    half_life_days: float,
) -> float:
    """Exponential evidence decay with a human-readable half-life."""
    age_seconds = max(0.0, (_as_utc(now) - _as_utc(observed_at)).total_seconds())
    age_days = age_seconds / 86400.0
    return math.exp(-math.log(2.0) * age_days / half_life_days)


def context_similarity(
    historical: ExperimentalContext,
    current: ExperimentalContext,
    *,
    weighting: ContextWeighting,
) -> float:
    """Return an explicit relevance weight in [0, 1] for old evidence."""
    if historical.target != current.target:
        return 0.0 if weighting.strict_target else weighting.mismatch_penalty

    weight = 1.0
    for field_name in (
        "assay",
        "protocol_version",
        "model_version",
        "reagent_lot",
        "instrument",
    ):
        old = getattr(historical, field_name)
        new = getattr(current, field_name)
        if old is None and new is None:
            continue
        if old is None or new is None:
            weight *= weighting.unknown_penalty
        elif old != new:
            weight *= weighting.mismatch_penalty
    return weight


def build_contextual_posteriors(
    observations: Iterable[ExperimentObservation],
    *,
    current_context: ExperimentalContext,
    now: datetime | None = None,
    prior_weight: float = 8.0,
    reference_rate: float = REFERENCE_BASE_RATE,
    weighting: ContextWeighting | None = None,
) -> dict[str, ContextualMethodPosterior]:
    """Build method posteriors from context- and time-weighted evidence.

    Only confirmed biological outcomes update binder probability. QC failures,
    technical failures, and missing binding calls remain observable events but
    do not get silently converted into biological negatives.

    Missing method provenance is deliberately not learned as a reusable method.
    """
    if prior_weight <= 0:
        raise ValueError("prior_weight must be positive")
    if not 0.0 < reference_rate < 1.0:
        raise ValueError("reference_rate must be between 0 and 1")

    weighting = weighting or ContextWeighting()
    now = _as_utc(now or datetime.now(timezone.utc))
    evidence: dict[str, list[float]] = {}
    raw_counts: dict[str, list[int]] = {}

    for observation in observations:
        if not observation.is_biological_evidence:
            continue
        method = observation.method or UNKNOWN_METHOD
        if method == UNKNOWN_METHOD:
            continue

        relevance = context_similarity(
            observation.context,
            current_context,
            weighting=weighting,
        )
        if relevance <= 0:
            continue
        weight = relevance * recency_weight(
            observation.observed_at,
            now=now,
            half_life_days=weighting.half_life_days,
        )
        if weight <= 0:
            continue

        eff = evidence.setdefault(method, [0.0, 0.0])
        raw = raw_counts.setdefault(method, [0, 0])
        eff[0] += weight
        eff[1] += weight * int(bool(observation.hit))
        raw[0] += 1
        raw[1] += int(bool(observation.hit))

    a0 = prior_weight * reference_rate
    b0 = prior_weight * (1.0 - reference_rate)
    out: dict[str, ContextualMethodPosterior] = {}

    for method, (effective_tested, effective_hits) in evidence.items():
        tested, hits = raw_counts[method]
        out[method] = ContextualMethodPosterior(
            method=method,
            alpha=a0 + effective_hits,
            beta=b0 + effective_tested - effective_hits,
            tested=tested,
            hits=hits,
            effective_tested=effective_tested,
            effective_hits=effective_hits,
        )

    out[UNKNOWN_METHOD] = ContextualMethodPosterior(
        method=UNKNOWN_METHOD,
        alpha=a0,
        beta=b0,
    )
    return out
