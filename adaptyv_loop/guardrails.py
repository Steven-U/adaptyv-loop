"""Spend guardrails for automated experiment submission.

The Foundry API exposes ``auto_accept_quote``, which finalizes a Stripe quote,
creates an invoice, and advances an experiment to "waiting for materials" with
no human in the loop. That is the right primitive for a design-test-learn loop
and it is also an agent holding a company credit card and a slice of finite
lab capacity.

Three controls sit between a campaign and that flag:

1. **A spend ceiling** per submission and per campaign, enforced against the
   API's own ``cost-estimate`` rather than against our guess at the price.
2. **A dry-run gate.** Every submission must first pass a non-billable
   rehearsal — spec validation plus a complete cost estimate. Live submission
   of a spec that has not passed is refused.
3. **An append-only decision log.** Every approval and every refusal is
   written to JSONL before the billable call is made, so the ledger cannot
   be missing a charge that actually happened.

Ordering matters: the log entry is flushed *before* the state-changing call.
Crashing mid-submission leaves an "attempted" record rather than silence.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .client import CostEstimate
from .errors import GuardrailViolation


@dataclass(frozen=True)
class SpendPolicy:
    """Limits a campaign may not exceed.

    ``max_campaign_usd`` is the number that matters: it bounds total spend
    across every submission the campaign makes, so a loop that misbehaves and
    submits repeatedly still cannot exceed it.
    """

    max_experiment_usd: float
    max_campaign_usd: float
    #: Live submission requires a matching successful dry run first.
    require_dry_run: bool = True
    #: Must be explicitly enabled before ``auto_accept_quote`` may be set.
    allow_auto_accept: bool = False
    #: Refuse when Foundry cannot price the experiment (unpriced target).
    require_complete_estimate: bool = True

    def __post_init__(self) -> None:
        if self.max_experiment_usd <= 0 or self.max_campaign_usd <= 0:
            raise ValueError("spend limits must be positive")
        if self.max_experiment_usd > self.max_campaign_usd:
            raise ValueError("max_experiment_usd cannot exceed max_campaign_usd")


@dataclass
class Decision:
    """One guardrail verdict, as written to the log."""

    timestamp: float
    action: str
    approved: bool
    reason: str
    estimated_usd: float | None = None
    campaign_spent_usd: float = 0.0
    campaign_id: str | None = None
    experiment_name: str | None = None
    n_sequences: int | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)


class DecisionLog:
    """Append-only JSONL log of guardrail decisions.

    Opened in append mode and flushed on every write. It is never rewritten or
    truncated, so it can be replayed to reconstruct what a campaign spent.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, decision: Decision) -> None:
        line = json.dumps(asdict(decision), sort_keys=True, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def replay(self) -> list[Decision]:
        if not self.path.exists():
            return []
        out = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                out.append(Decision(**json.loads(raw)))
        return out

    def total_approved_usd(self) -> float:
        """Sum of committed spend.

        Only ``approve`` entries count. A passing ``dry_run`` carries the same
        estimate but commits nothing, so including it would double-count every
        submission and halve the effective budget on resume.
        """
        return sum(
            d.estimated_usd or 0.0
            for d in self.replay()
            if d.approved and d.action == "approve"
        )


class SpendGuard:
    """Enforces a :class:`SpendPolicy` and records every verdict.

    Usage is deliberately two-step — :meth:`dry_run` then :meth:`approve` —
    so that the expensive call cannot be reached without the free one having
    succeeded on the same spec.
    """

    def __init__(
        self,
        policy: SpendPolicy,
        log: DecisionLog,
        *,
        campaign_id: str | None = None,
    ) -> None:
        self.policy = policy
        self.log = log
        self.campaign_id = campaign_id
        self._lock = threading.Lock()
        # Resume from the log so a restarted campaign does not get a fresh budget.
        self._spent_usd = log.total_approved_usd()
        self._dry_run_passed: dict[str, float] = {}

    @property
    def spent_usd(self) -> float:
        return self._spent_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.policy.max_campaign_usd - self._spent_usd)

    @staticmethod
    def spec_fingerprint(spec: Mapping[str, Any]) -> str:
        """Stable hash of an experiment spec, tying a dry run to its submission."""
        import hashlib

        blob = json.dumps(spec, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def _record(self, **kwargs: Any) -> Decision:
        decision = Decision(
            timestamp=time.time(),
            campaign_id=self.campaign_id,
            campaign_spent_usd=self._spent_usd,
            **kwargs,
        )
        self.log.write(decision)
        return decision

    def dry_run(
        self,
        spec: Mapping[str, Any],
        estimate: CostEstimate,
        *,
        experiment_name: str | None = None,
    ) -> Decision:
        """Validate a priced spec without spending anything.

        Records the verdict and, on success, remembers the fingerprint so
        :meth:`approve` will accept this exact spec.
        """
        fingerprint = self.spec_fingerprint(spec)
        n_seq = len(spec.get("sequences") or {})
        common = {
            "action": "dry_run",
            "experiment_name": experiment_name,
            "n_sequences": n_seq,
            "estimated_usd": estimate.total_usd,
            "detail": {"fingerprint": fingerprint, "warnings": list(estimate.warnings)},
        }

        if self.policy.require_complete_estimate and not estimate.is_complete:
            decision = self._record(
                approved=False,
                reason=(
                    "Foundry returned an incomplete cost estimate (target likely "
                    "lacks self-service pricing). Unknown cost is not zero cost."
                ),
                **common,
            )
            raise GuardrailViolation(decision.reason)

        cost = estimate.total_usd or 0.0
        if cost > self.policy.max_experiment_usd:
            decision = self._record(
                approved=False,
                reason=(
                    f"${cost:,.2f} exceeds per-experiment ceiling of "
                    f"${self.policy.max_experiment_usd:,.2f}"
                ),
                **common,
            )
            raise GuardrailViolation(decision.reason)

        if cost > self.remaining_usd:
            decision = self._record(
                approved=False,
                reason=(
                    f"${cost:,.2f} exceeds remaining campaign budget of "
                    f"${self.remaining_usd:,.2f} "
                    f"(spent ${self._spent_usd:,.2f} of ${self.policy.max_campaign_usd:,.2f})"
                ),
                **common,
            )
            raise GuardrailViolation(decision.reason)

        with self._lock:
            self._dry_run_passed[fingerprint] = cost
        return self._record(approved=True, reason="dry run passed", **common)

    def approve(
        self,
        spec: Mapping[str, Any],
        *,
        experiment_name: str | None = None,
        auto_accept_quote: bool = False,
    ) -> Decision:
        """Authorize a billable submission, committing the estimate to the budget.

        Call immediately before the state-changing API call. Raises
        :class:`GuardrailViolation` if any control fails, in which case no
        budget is consumed and nothing has been submitted.
        """
        fingerprint = self.spec_fingerprint(spec)
        n_seq = len(spec.get("sequences") or {})
        common = {
            "action": "approve",
            "experiment_name": experiment_name,
            "n_sequences": n_seq,
            "detail": {"fingerprint": fingerprint, "auto_accept_quote": auto_accept_quote},
        }

        if auto_accept_quote and not self.policy.allow_auto_accept:
            decision = self._record(
                approved=False,
                estimated_usd=None,
                reason=(
                    "auto_accept_quote requires SpendPolicy(allow_auto_accept=True). "
                    "This flag creates an invoice with no human confirmation."
                ),
                **common,
            )
            raise GuardrailViolation(decision.reason)

        with self._lock:
            cost = self._dry_run_passed.get(fingerprint)

            if self.policy.require_dry_run and cost is None:
                decision = self._record(
                    approved=False,
                    estimated_usd=None,
                    reason=(
                        f"no passing dry run for spec {fingerprint}. Call dry_run() "
                        "with this exact spec first; any edit changes the fingerprint."
                    ),
                    **common,
                )
                raise GuardrailViolation(decision.reason)

            cost = cost or 0.0
            # Re-check against the live budget: another submission may have
            # consumed it between this spec's dry run and now.
            if cost > self.remaining_usd:
                decision = self._record(
                    approved=False,
                    estimated_usd=cost,
                    reason=(
                        f"${cost:,.2f} exceeds remaining budget of "
                        f"${self.remaining_usd:,.2f} at approval time"
                    ),
                    **common,
                )
                raise GuardrailViolation(decision.reason)

            self._spent_usd += cost
            self._dry_run_passed.pop(fingerprint, None)

        return self._record(
            approved=True,
            estimated_usd=cost,
            reason="within policy; budget committed",
            **common,
        )
