"""End-to-end demo with no API token and no spend.

Runs the full design-test-learn loop — select, price, guardrail, submit,
collect, re-rank — against a simulated lab whose answers are the *real*
wet-lab outcomes from Adaptyv's public EGFR competition. Every binding call
the loop learns from actually happened at the bench.

    python demo_offline.py

The point is to watch the method posteriors move between rounds and to see
the guardrail refuse a round that breaches the budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "backtest"))

import pandas as pd  # noqa: E402

from adaptyv_loop import (  # noqa: E402
    Campaign,
    Candidate,
    CostEstimate,
    GuardrailViolation,
    SpendPolicy,
    campaign_report,
)
from run_backtest import ensure_data, load_round2  # noqa: E402

PRICE_PER_PROTEIN_CENTS = 9_900


class SimulatedLab:
    """Stands in for FoundryClient, answering from real competition outcomes."""

    def __init__(self, truth: pd.DataFrame) -> None:
        self.truth = truth.set_index("design_id")
        self.experiments: dict[str, list[str]] = {}
        self._n = 0

    def cost_estimate(self, spec) -> CostEstimate:
        n = len(spec["sequences"]) * (spec.get("n_replicates") or 1)
        return CostEstimate(total_cents=n * PRICE_PER_PROTEIN_CENTS, pricing_version="demo")

    def create_experiment(self, name, spec, *, auto_accept_quote=False, skip_draft=False):
        self._n += 1
        exp_id = f"exp-{self._n:03d}"
        self.experiments[exp_id] = list(spec["sequences"])
        return {"id": exp_id, "name": name, "status": "in_queue"}

    def submit_experiment(self, experiment_id):
        return {"id": experiment_id, "status": "in_production"}

    def get_results(self, experiment_id):
        names = self.experiments[experiment_id]
        return [
            {
                "summary": [
                    {
                        "sequence": {"name": n},
                        "binding": "true" if bool(self.truth.loc[n, "hit"]) else "false",
                    }
                    for n in names
                ]
            }
        ]

    def wait_for_results(self, experiment_id, **_):
        return self.get_results(experiment_id)


def main() -> int:
    data_dir = Path(__file__).parent / "backtest" / "data"
    ensure_data(data_dir)
    df = load_round2(data_dir).reset_index(drop=True)
    df["design_id"] = [f"d{i:03d}" for i in range(len(df))]

    candidates = [
        Candidate(id=r.design_id, sequence=str(r.sequence), method=r.method, score=float(r.iptm))
        for r in df.itertuples()
    ]

    lab = SimulatedLab(df[["design_id", "hit"]])
    campaign = Campaign(
        lab,  # type: ignore[arg-type]
        campaign_id="egfr-demo",
        target_id="demo-egfr-target",
        policy=SpendPolicy(
            max_experiment_usd=6_000,
            max_campaign_usd=14_000,
            allow_auto_accept=True,
        ),
        state_dir=Path(__file__).parent / ".adaptyv-demo",
        n_replicates=1,
    )

    print(f"Pool: {len(candidates)} designs across {df.method.nunique()} design methods")
    print(f"Budget: ${campaign.guard.policy.max_campaign_usd:,.0f} "
          f"at ${PRICE_PER_PROTEIN_CENTS / 100:.0f}/protein "
          f"= {campaign.guard.policy.max_campaign_usd * 100 // PRICE_PER_PROTEIN_CENTS:.0f} tests\n")

    for _ in range(3):
        print("=" * 70)
        # Numbered off campaign state, not the loop counter, so a resumed run
        # continues at round 4 rather than restarting at round 1.
        print(f"ROUND {len(campaign.state.rounds) + 1}")
        print("=" * 70)
        try:
            selection = campaign.plan_round(candidates, 45)
        except ValueError as exc:
            print(f"stopping: {exc}")
            break

        for line in selection.rationale:
            print(line)

        try:
            _, estimate = campaign.dry_run(selection)
            print(f"\ndry run OK — Foundry would bill ${estimate.total_usd:,.0f}")
        except GuardrailViolation as exc:
            print(f"\nGUARDRAIL REFUSED THE ROUND: {exc}")
            print("Nothing was submitted and no invoice exists.")
            break

        record = campaign.submit_round(selection, auto_accept_quote=True)
        campaign.collect_round(record, candidates)
        print(f"results: {record.hits}/{len(record.outcomes)} binders "
              f"({record.hits / max(len(record.outcomes), 1):.1%})\n")

    print("=" * 70)
    print(campaign_report(campaign))
    print("\nState and decision log written to .adaptyv-demo/")
    print("Re-running resumes the same campaign rather than starting a fresh budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
