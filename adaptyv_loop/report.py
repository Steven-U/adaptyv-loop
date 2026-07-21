"""Campaign reporting, in the units a budget holder thinks in."""

from __future__ import annotations

from .campaign import Campaign, CampaignState


def _rate(hits: int, tested: int) -> str:
    return f"{hits / tested:.1%}" if tested else "—"


def campaign_report(campaign: Campaign) -> str:
    """Render a plain-text summary of a campaign's spend and yield."""
    state: CampaignState = campaign.state
    guard = campaign.guard
    lines: list[str] = []

    lines.append(f"Campaign {state.campaign_id}")
    lines.append(f"target {state.target_id} | {state.experiment_type}"
                 + (f"/{state.method}" if state.method else ""))
    lines.append("")

    total_tested = sum(len(r.outcomes) for r in state.rounds)
    total_hits = sum(r.hits for r in state.rounds)
    spent = guard.spent_usd

    lines.append(f"{'round':<7}{'designs':>9}{'results':>9}{'binders':>9}{'rate':>8}{'cost':>11}")
    lines.append("-" * 53)
    for r in state.rounds:
        cost = f"${r.estimated_usd:,.0f}" if r.estimated_usd else "—"
        lines.append(
            f"{r.index:<7}{r.n_submitted:>9}{len(r.outcomes):>9}{r.hits:>9}"
            f"{_rate(r.hits, len(r.outcomes)):>8}{cost:>11}"
        )
    lines.append("-" * 53)
    lines.append(
        f"{'total':<7}{sum(r.n_submitted for r in state.rounds):>9}{total_tested:>9}"
        f"{total_hits:>9}{_rate(total_hits, total_tested):>8}{f'${spent:,.0f}':>11}"
    )
    lines.append("")

    if total_hits:
        lines.append(f"Cost per binder: ${spent / total_hits:,.0f}")
    lines.append(
        f"Budget: ${spent:,.0f} of ${guard.policy.max_campaign_usd:,.0f} "
        f"(${guard.remaining_usd:,.0f} left)"
    )
    lines.append("")

    if state.method_history:
        lines.append("Method performance (drives the next round's allocation)")
        lines.append(f"{'method':<44}{'tested':>8}{'binders':>9}{'rate':>8}")
        lines.append("-" * 69)
        ranked = sorted(
            state.method_history.items(),
            key=lambda kv: -(kv[1][1] / kv[1][0] if kv[1][0] else 0),
        )
        for method, (tested, hits) in ranked:
            lines.append(f"{method[:43]:<44}{tested:>8}{hits:>9}{_rate(hits, tested):>8}")

    refused = [d for d in guard.log.replay() if not d.approved]
    if refused:
        lines.append("")
        lines.append(f"Guardrail refusals: {len(refused)}")
        for d in refused[-5:]:
            lines.append(f"  {d.action}: {d.reason}")

    return "\n".join(lines)
