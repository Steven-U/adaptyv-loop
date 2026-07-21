"""adaptyv-loop — a budgeted design-test-learn loop over the Adaptyv Foundry API.

    from adaptyv_loop import Campaign, Candidate, FoundryClient, SpendPolicy

    client = FoundryClient()                      # reads ADAPTYV_API_TOKEN
    campaign = Campaign(
        client,
        campaign_id="egfr-q3",
        target_id=egfr_target_id,
        policy=SpendPolicy(max_experiment_usd=5_000, max_campaign_usd=20_000),
    )
    selection = campaign.plan_round(candidates, n_slots=48)
    campaign.dry_run(selection)                   # free: prices and validates
    campaign.submit_round(selection)              # bills, after guardrails pass
"""

from .campaign import Campaign, CampaignState, RoundRecord, outcomes_from_results
from .client import CostEstimate, FoundryClient, build_experiment_spec
from .errors import (
    AdaptyvError,
    APIError,
    AuthError,
    GuardrailViolation,
    RateLimited,
)
from .guardrails import Decision, DecisionLog, SpendGuard, SpendPolicy
from .report import campaign_report
from .selection import (
    Candidate,
    MethodPosterior,
    MethodStats,
    Selection,
    build_posteriors,
    select_designs,
)

__version__ = "0.1.0"

__all__ = [
    "APIError",
    "AdaptyvError",
    "AuthError",
    "Campaign",
    "CampaignState",
    "Candidate",
    "CostEstimate",
    "Decision",
    "DecisionLog",
    "FoundryClient",
    "GuardrailViolation",
    "MethodPosterior",
    "MethodStats",
    "RateLimited",
    "RoundRecord",
    "Selection",
    "SpendGuard",
    "SpendPolicy",
    "build_experiment_spec",
    "build_posteriors",
    "campaign_report",
    "outcomes_from_results",
    "select_designs",
]
