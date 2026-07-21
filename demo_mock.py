"""The full design-test-learn loop over real HTTP, validated against Adaptyv's
real published OpenAPI contract. No API token, no spend, no access required.

Topology (started by ``mock/serve.sh``):

    demo_mock.py -> Prism (:4010) -> mock Foundry (:4011)
                     |                  |
                     |                  realistic data: real EGFR target,
                     |                  $99/protein pricing, real wet-lab
                     |                  binding outcomes from the public
                     |                  competition
                     |
                     validates every request AND response against Adaptyv's
                     genuine OpenAPI spec (backtest/foundry_openapi.json).
                     A non-conforming payload is a 4xx, not a silent pass.

So this exercises the real client code path — whoami, catalog lookup, cost
estimate, create, submit, poll, results, re-rank — against the genuine schema,
with none of the placeholder data a hand-written mock would invent.

    ./mock/serve.sh            # terminal 1: bring up the validated stack
    python demo_mock.py        # terminal 2: run the loop
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "backtest"))

import pandas as pd  # noqa: E402

from adaptyv_loop import (  # noqa: E402
    Campaign,
    Candidate,
    FoundryClient,
    GuardrailViolation,
    SpendPolicy,
    build_experiment_spec,
    campaign_report,
)
from run_backtest import ensure_data, load_round2  # noqa: E402

PROXY = "http://127.0.0.1:4010"


def preflight() -> bool:
    try:
        requests.get(f"{PROXY}/api/v1/whoami", timeout=3)
        return True
    except requests.RequestException:
        print("The validated mock stack is not running. In another terminal:\n")
        print("    ./mock/serve.sh\n")
        print("That starts the mock Foundry behind a Prism proxy which validates")
        print("every call against Adaptyv's real OpenAPI spec. Then re-run this.")
        return False


def build_pool() -> tuple[list[Candidate], pd.DataFrame]:
    data_dir = Path(__file__).parent / "backtest" / "data"
    ensure_data(data_dir)
    df = load_round2(data_dir).reset_index(drop=True)
    df["design_id"] = [f"d{i:03d}" for i in range(len(df))]
    pool = [
        Candidate(id=r.design_id, sequence=str(r.sequence), method=r.method, score=float(r.iptm))
        for r in df.itertuples()
    ]
    return pool, df


def step(n: int, title: str) -> None:
    print(f"\n{'=' * 70}\n{n}. {title}\n{'=' * 70}")


def main() -> int:
    if not preflight():
        return 1

    # The client cannot tell this from the production API: same code, same
    # protocol. Only the base URL points at the validating proxy.
    client = FoundryClient(token="demo-token", base_url=PROXY)
    pool, df = build_pool()

    step(1, "Authenticate (validated: WhoAmIResponse)")
    me = client.whoami()
    print(f"org {me['active_organization_id']} | permissions {me['permissions']}")

    step(2, "Resolve the EGFR target from the catalog (validated: TargetInfo)")
    target = client.find_target("EGFR")[0]
    price = (target.get("pricing") or {}).get("price_per_sequence_cents", 9900) / 100
    print(f"{target['name']}")
    print(f"  id {target['id']} | UniProt {target['uniprot_id']} | ${price:.0f}/protein")

    step(3, "Local rule check that Prism cannot do")
    print("Prism validates the structural schema — it 422s a spec missing")
    print("experiment_type. But the type/field matrix is documented in prose in")
    print("the spec, so the client enforces it before the request is ever sent:")
    for label, kwargs in [
        ("thermostability + target_id", dict(experiment_type="thermostability",
                                             sequences={"a": "MKT"}, target_id=target["id"])),
        ("screening without a method", dict(experiment_type="screening",
                                            sequences={"a": "MKT"}, target_id=target["id"])),
    ]:
        try:
            build_experiment_spec(**kwargs)
            print(f"  {label}: unexpectedly accepted")
        except ValueError as exc:
            print(f"  {label}: rejected — {exc}")

    step(4, "Run the design-test-learn loop through the validated proxy")
    campaign = Campaign(
        client,
        campaign_id="egfr-mock",
        target_id=target["id"],
        policy=SpendPolicy(max_experiment_usd=6_000, max_campaign_usd=14_000),
        state_dir=Path(__file__).parent / ".adaptyv-mock",
        n_replicates=1,
    )
    print(f"pool {len(pool)} designs | budget ${campaign.guard.policy.max_campaign_usd:,.0f}\n")

    for _ in range(3):
        rnd = len(campaign.state.rounds) + 1
        try:
            selection = campaign.plan_round(pool, 45)
        except ValueError as exc:
            print(f"round {rnd}: {exc}")
            break
        try:
            _, estimate = campaign.dry_run(selection)
        except GuardrailViolation as exc:
            print(f"round {rnd}: GUARDRAIL REFUSED — {exc}")
            break
        # create + submit + poll + results, every hop validated by Prism
        record = campaign.submit_round(selection)
        campaign.collect_round(record, pool)
        top = max(selection.per_method, key=selection.per_method.get)
        print(f"round {rnd}: {record.n_submitted} designs "
              f"(${estimate.total_usd:,.0f}, top method {top}) "
              f"-> {record.hits}/{len(record.outcomes)} binders "
              f"({record.hits / max(len(record.outcomes), 1):.0%})")

    step(5, "Campaign report")
    print(campaign_report(campaign))
    print("\nEvery HTTP call above was validated against Adaptyv's real OpenAPI")
    print("spec by the Prism proxy. Nothing here required API access.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
