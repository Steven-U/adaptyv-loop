"""Live demo against the real Adaptyv Foundry API. Spends nothing.

Every call this makes is non-billable: ``/whoami``, ``/targets``, and
``/experiments/cost-estimate``. Nothing is created and no quote is accepted,
so it is safe to run against a production token.

    export ADAPTYV_API_TOKEN=...
    python demo_live.py --target EGFR --slots 48

What it shows, end to end:

  1. Authenticate and resolve the token's org.
  2. Search the live target catalog and read back real pricing.
  3. Pick designs under a budget using method history.
  4. Ask Foundry what that experiment would actually cost.
  5. Run it past the guardrails, then deliberately breach one so you can see
     a refusal happen before any state-changing call.

The last submit step is printed rather than executed. Pass --i-want-to-spend-money
to actually create the experiment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from adaptyv_loop import (  # noqa: E402
    AuthError,
    Candidate,
    FoundryClient,
    GuardrailViolation,
    MethodStats,
    SpendPolicy,
    build_experiment_spec,
    build_posteriors,
    select_designs,
)
from adaptyv_loop.guardrails import DecisionLog, SpendGuard  # noqa: E402

# Stand-in candidate pool. Swap for your own designs; `method` is whatever
# generated the sequence and is the field that drives allocation.
DEMO_POOL = [
    ("ProteinMPNN/LigandMPNN", 0.94, 12),
    ("Custom PLM", 0.91, 14),
    ("BindCraft", 0.89, 16),
    ("ProteinMPNN/LigandMPNN + RFdiffusion", 0.93, 20),
    ("ESM2/3", 0.85, 10),
]
# Method history from a hypothetical prior campaign — in production this comes
# from CampaignState, accumulated across every campaign you have run.
PRIOR_HISTORY = [
    MethodStats("ProteinMPNN/LigandMPNN", tested=30, hits=13),
    MethodStats("Custom PLM", tested=57, hits=14),
    MethodStats("BindCraft", tested=49, hits=6),
    MethodStats("ProteinMPNN/LigandMPNN + RFdiffusion", tested=83, hits=6),
    MethodStats("ESM2/3", tested=16, hits=1),
]

SCFV = (
    "EVQLLESGGGVVKPGGSLRLSCAASGRTFSSYAMGWFRQAPGKGLEWVSAINWSSGSTYYADSVKG"
    "RFTISRDDAKNSLYLQMNSLRAEDTAVYYCAAGYQINSGNYNFKDYEYDYWGQGTLVTVSS"
)


def build_pool() -> list[Candidate]:
    out: list[Candidate] = []
    for method, base_score, n in DEMO_POOL:
        for i in range(n):
            out.append(
                Candidate(
                    id=f"{method.split()[0][:8]}-{i:02d}".replace("/", "_"),
                    sequence=SCFV,
                    method=method,
                    score=round(base_score - i * 0.005, 3),
                )
            )
    return out


def step(n: int, title: str) -> None:
    print(f"\n{'=' * 68}\n{n}. {title}\n{'=' * 68}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="EGFR", help="catalog search term")
    ap.add_argument("--slots", type=int, default=48)
    ap.add_argument("--budget", type=float, default=20_000.0)
    ap.add_argument(
        "--i-want-to-spend-money",
        action="store_true",
        help="actually create the experiment (BILLS THE ACCOUNT)",
    )
    args = ap.parse_args()

    try:
        client = FoundryClient()
    except AuthError as exc:
        print(f"{exc}\n\nGet a token from the Adaptyv Portal, then:")
        print("  export ADAPTYV_API_TOKEN=...")
        return 2

    step(1, "Authenticate")
    try:
        me = client.whoami()
    except AuthError as exc:
        print(f"{exc}\n\nThe API is reachable but rejected this token.")
        return 2
    print(f"token resolves to: {me}")

    step(2, "Resolve the target from the live catalog")
    matches = client.find_target(args.target)
    if not matches:
        print(f"no catalog target matched {args.target!r}.")
        print("Run with --target '' to list everything, or request a custom target.")
        return 1
    for t in matches[:5]:
        pricing = t.get("pricing") or {}
        print(f"  {t['id']}  {t['name']}  ({t.get('vendor_name')})  pricing={pricing or 'n/a'}")
    target = matches[0]
    print(f"\nusing: {target['name']} [{target['id']}]")

    step(3, "Select designs under budget, using prior-campaign method history")
    pool = build_pool()
    selection = select_designs(
        pool, args.slots, posteriors=build_posteriors(PRIOR_HISTORY), mode="exploit"
    )
    print(f"pool: {len(pool)} designs\n")
    for line in selection.rationale:
        print(line)

    step(4, "Ask Foundry what this would actually cost (free call)")
    spec = build_experiment_spec(
        experiment_type="screening",
        sequences=selection.sequences,
        target_id=target["id"],
        method="bli",
        n_replicates=2,
    )
    estimate = client.cost_estimate(spec)
    if estimate.is_complete:
        print(f"Foundry quote: ${estimate.total_usd:,.2f} "
              f"for {len(selection)} designs (pricing {estimate.pricing_version})")
        print(f"              ${estimate.total_usd / len(selection):,.2f} per design")
    else:
        print("Foundry returned an INCOMPLETE estimate — this target has no "
              "self-service pricing.")
    for w in estimate.warnings:
        print(f"  warning: {w}")

    step(5, "Guardrails")
    guard = SpendGuard(
        SpendPolicy(max_experiment_usd=args.budget, max_campaign_usd=args.budget),
        DecisionLog(Path(".adaptyv-live") / "decisions.jsonl"),
        campaign_id="live-demo",
    )
    try:
        guard.dry_run(spec, estimate, experiment_name="live-demo-r1")
        print(f"dry run PASSED — ${estimate.total_usd or 0:,.2f} is within "
              f"the ${args.budget:,.0f} ceiling")
    except GuardrailViolation as exc:
        print(f"dry run REFUSED: {exc}")
        return 0

    print("\nNow the same experiment against a $100 ceiling:")
    tight = SpendGuard(
        SpendPolicy(max_experiment_usd=100, max_campaign_usd=100),
        DecisionLog(Path(".adaptyv-live") / "decisions.jsonl"),
        campaign_id="live-demo-tight",
    )
    try:
        tight.dry_run(spec, estimate, experiment_name="live-demo-tight")
        print("  unexpectedly passed")
    except GuardrailViolation as exc:
        print(f"  REFUSED: {exc}")

    print("\nAnd auto_accept_quote without arming the policy:")
    try:
        guard.approve(spec, experiment_name="live-demo-r1", auto_accept_quote=True)
        print("  unexpectedly passed")
    except GuardrailViolation as exc:
        print(f"  REFUSED: {exc}")

    step(6, "Submit")
    if not args.i_want_to_spend_money:
        print("Not submitting. This is where the loop would call:")
        print(f"  POST /experiments            name=live-demo-r1, {len(selection)} sequences")
        print("  POST /experiments/{id}/submit")
        print(f"  ... then poll /results and fold {len(selection)} outcomes back into")
        print("      the method posteriors for the next round.")
        print(f"\nRe-run with --i-want-to-spend-money to bill "
              f"${estimate.total_usd or 0:,.2f} for real.")
        return 0

    guard.approve(spec, experiment_name="live-demo-r1")
    experiment = client.create_experiment("live-demo-r1", spec)
    client.submit_experiment(experiment["id"])
    print(f"submitted: {experiment['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
