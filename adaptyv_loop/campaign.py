"""The design-test-learn loop.

Ties the three pieces together: :mod:`~adaptyv_loop.selection` decides what to
buy, :mod:`~adaptyv_loop.guardrails` decides whether it is allowed to, and
:mod:`~adaptyv_loop.client` submits it and pulls results back.

The loop is deliberately resumable. Lab turnaround is on the order of three
weeks, so a campaign object will outlive the process that created it: state is
persisted after every round and reloaded on construction. Restarting mid-flight
picks up the same budget, the same history, and the same in-flight experiments
rather than starting a fresh campaign against the same credit card.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .client import ExperimentType, FoundryClient, Method, build_experiment_spec
from .errors import APIError, GuardrailViolation
from .guardrails import DecisionLog, SpendGuard, SpendPolicy
from .selection import (
    Candidate,
    MethodPosterior,
    MethodStats,
    Selection,
    build_posteriors,
    select_designs,
)

log = logging.getLogger(__name__)


@dataclass
class RoundRecord:
    """What one round of the loop bought and what came back."""

    index: int
    experiment_id: str | None
    experiment_name: str
    n_submitted: int
    estimated_usd: float | None
    per_method: dict[str, int] = field(default_factory=dict)
    candidate_ids: list[str] = field(default_factory=list)
    #: candidate id -> observed binder / non-binder, once results land.
    outcomes: dict[str, bool] = field(default_factory=dict)
    status: str = "submitted"

    @property
    def hits(self) -> int:
        return sum(1 for v in self.outcomes.values() if v)


@dataclass
class CampaignState:
    """Everything needed to resume a campaign."""

    campaign_id: str
    target_id: str
    experiment_type: str
    method: str | None
    rounds: list[RoundRecord] = field(default_factory=list)
    #: method -> [tested, hits], accumulated across rounds and prior campaigns.
    method_history: dict[str, list[int]] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, blob: str) -> CampaignState:
        raw = json.loads(blob)
        rounds = [RoundRecord(**r) for r in raw.pop("rounds", [])]
        return cls(rounds=rounds, **raw)

    def stats(self) -> list[MethodStats]:
        return [
            MethodStats(method=m, tested=t, hits=h)
            for m, (t, h) in self.method_history.items()
            if t > 0
        ]

    def record_outcomes(self, method_by_id: Mapping[str, str], outcomes: Mapping[str, bool]) -> None:
        for cid, hit in outcomes.items():
            method = method_by_id.get(cid, "unknown")
            rec = self.method_history.setdefault(method, [0, 0])
            rec[0] += 1
            rec[1] += int(hit)


def outcomes_from_results(results: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    """Flatten Foundry result payloads into ``{sequence_name: is_binder}``.

    Real result payloads are messy: ``binding`` rolls up to the strings
    "true"/"false"/"unknown", and a sequence whose construct failed to express
    has no meaningful binding call at all. Unknowns are dropped rather than
    counted as non-binders — a failed expression is missing data, and scoring
    it as a miss would slowly poison the method posteriors that drive
    selection.
    """
    out: dict[str, bool] = {}
    for result in results:
        for entry in result.get("summary") or []:
            seq = entry.get("sequence") or {}
            name = seq.get("name") or entry.get("name")
            if not name:
                continue
            binding = str(entry.get("binding") or "").lower()
            if binding in ("true", "false"):
                out[name] = binding == "true"
                continue
            # Fall back to the categorical call when `binding` is absent.
            strength = str(entry.get("binding_strength") or "").lower()
            if strength in ("strong", "medium", "weak"):
                out[name] = strength != "none"
            elif strength == "none":
                out[name] = False
    return out


class Campaign:
    """A budgeted, resumable design-test-learn campaign against one target."""

    def __init__(
        self,
        client: FoundryClient,
        *,
        campaign_id: str,
        target_id: str,
        policy: SpendPolicy,
        experiment_type: ExperimentType = "screening",
        method: Method | None = "bli",
        state_dir: str | Path = ".adaptyv",
        n_replicates: int | None = 2,
    ) -> None:
        self.client = client
        self.n_replicates = n_replicates
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.state_dir / f"{campaign_id}.state.json"

        self.guard = SpendGuard(
            policy,
            DecisionLog(self.state_dir / f"{campaign_id}.decisions.jsonl"),
            campaign_id=campaign_id,
        )

        if self._state_path.exists():
            self.state = CampaignState.from_json(self._state_path.read_text())
            log.info(
                "resumed campaign %s: %d rounds, $%.2f spent",
                campaign_id,
                len(self.state.rounds),
                self.guard.spent_usd,
            )
        else:
            self.state = CampaignState(
                campaign_id=campaign_id,
                target_id=target_id,
                experiment_type=experiment_type,
                method=method,
            )
            self._save()

    def _save(self) -> None:
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(self.state.to_json())
        tmp.replace(self._state_path)  # atomic, so a crash cannot truncate state

    # ---- planning ------------------------------------------------------

    def plan_round(
        self,
        candidates: Sequence[Candidate],
        n_slots: int,
        *,
        mode: str | None = None,
        posteriors: Mapping[str, MethodPosterior] | None = None,
        **kwargs: Any,
    ) -> Selection:
        """Choose designs for the next round using accumulated history.

        Defaults to ``explore`` while history is thin and ``exploit`` once
        there is enough to separate methods. A caller may inject explicit
        posteriors, including context-aware external posteriors, without
        changing the campaign's execution or spend guardrails.
        """
        history = self.state.stats()
        tested = sum(s.tested for s in history)
        if mode is None:
            mode = "exploit" if tested >= 30 else "explore"
        already = {cid for r in self.state.rounds for cid in r.candidate_ids}
        fresh = [c for c in candidates if c.id not in already]
        if not fresh:
            raise ValueError("every candidate has already been submitted in a prior round")
        return select_designs(
            fresh,
            n_slots,
            posteriors=posteriors if posteriors is not None else build_posteriors(history),
            mode=mode,  # type: ignore[arg-type]
            **kwargs,
        )

    def _spec(self, selection: Selection) -> dict[str, Any]:
        return build_experiment_spec(
            experiment_type=self.state.experiment_type,  # type: ignore[arg-type]
            sequences=selection.sequences,
            target_id=self.state.target_id,
            method=self.state.method,  # type: ignore[arg-type]
            n_replicates=self.n_replicates,
        )

    def dry_run(self, selection: Selection) -> tuple[dict[str, Any], Any]:
        """Price and validate a round without spending anything.

        Returns the spec and the cost estimate. Raises
        :class:`~.errors.GuardrailViolation` if the round would breach policy,
        which is the point: you learn that from a free call.
        """
        spec = self._spec(selection)
        estimate = self.client.cost_estimate(spec)
        name = f"{self.state.campaign_id}-r{len(self.state.rounds) + 1}"
        self.guard.dry_run(spec, estimate, experiment_name=name)
        return spec, estimate

    # ---- execution -----------------------------------------------------

    def submit_round(
        self,
        selection: Selection,
        *,
        auto_accept_quote: bool = False,
    ) -> RoundRecord:
        """Price, authorize, and submit a round. **Bills the account.**

        Runs the dry run itself so the guardrail's fingerprint check cannot be
        bypassed by submitting a spec that was never priced.
        """
        spec, estimate = self.dry_run(selection)
        name = f"{self.state.campaign_id}-r{len(self.state.rounds) + 1}"

        decision = self.guard.approve(
            spec, experiment_name=name, auto_accept_quote=auto_accept_quote
        )

        experiment = self.client.create_experiment(
            name, spec, auto_accept_quote=auto_accept_quote
        )
        # CreateExpResponse keys the id as `experiment_id`; ExpInfo and most
        # other reads key it as `id`. Accept either so this does not depend on
        # which endpoint shape a given response happens to use.
        experiment_id = experiment.get("experiment_id") or experiment.get("id")
        if not experiment_id:
            raise APIError(0, f"create_experiment returned no id: {experiment}")
        if not auto_accept_quote:
            # Without auto-accept the experiment sits in draft until a human
            # confirms the quote; submit advances it as far as it can go.
            self.client.submit_experiment(experiment_id)

        record = RoundRecord(
            index=len(self.state.rounds) + 1,
            experiment_id=experiment_id,
            experiment_name=name,
            n_submitted=len(selection),
            estimated_usd=decision.estimated_usd,
            per_method=dict(selection.per_method),
            candidate_ids=[c.id for c in selection.chosen],
        )
        self.state.rounds.append(record)
        self._save()
        log.info(
            "round %d submitted: %d designs, $%.2f, experiment %s",
            record.index,
            record.n_submitted,
            record.estimated_usd or 0.0,
            experiment_id,
        )
        return record

    def collect_round(
        self,
        record: RoundRecord,
        candidates: Sequence[Candidate],
        *,
        wait: bool = True,
        **wait_kwargs: Any,
    ) -> RoundRecord:
        """Pull results for a round and fold them into the method history."""
        if not record.experiment_id:
            raise ValueError("round has no experiment id")
        results = (
            self.client.wait_for_results(record.experiment_id, **wait_kwargs)
            if wait
            else self.client.get_results(record.experiment_id)
        )
        outcomes = outcomes_from_results(results)
        submitted = set(record.candidate_ids)
        outcomes = {k: v for k, v in outcomes.items() if k in submitted}

        record.outcomes = outcomes
        record.status = "done"
        method_by_id = {c.id: c.method for c in candidates}
        self.state.record_outcomes(method_by_id, outcomes)
        self._save()
        log.info(
            "round %d results: %d/%d binders",
            record.index,
            record.hits,
            len(outcomes),
        )
        return record

    def run_round(
        self,
        candidates: Sequence[Candidate],
        n_slots: int,
        *,
        auto_accept_quote: bool = False,
        wait: bool = True,
        **wait_kwargs: Any,
    ) -> RoundRecord:
        """Plan, submit, and collect one full round."""
        selection = self.plan_round(candidates, n_slots)
        for line in selection.rationale:
            log.info("%s", line)
        record = self.submit_round(selection, auto_accept_quote=auto_accept_quote)
        return self.collect_round(record, candidates, wait=wait, **wait_kwargs)

    def run(
        self,
        candidates: Sequence[Candidate],
        *,
        slots_per_round: int,
        max_rounds: int = 3,
        auto_accept_quote: bool = False,
    ) -> list[RoundRecord]:
        """Run rounds until the budget, the candidate pool, or ``max_rounds`` runs out."""
        done: list[RoundRecord] = []
        for _ in range(max_rounds):
            if self.guard.remaining_usd <= 0:
                log.info("campaign budget exhausted; stopping")
                break
            try:
                done.append(
                    self.run_round(
                        candidates, slots_per_round, auto_accept_quote=auto_accept_quote
                    )
                )
            except GuardrailViolation as exc:
                log.warning("stopping: %s", exc)
                break
            except ValueError as exc:
                log.info("stopping: %s", exc)
                break
        return done
