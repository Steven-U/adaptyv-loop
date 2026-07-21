"""Client for the Adaptyv Foundry API.

Wraps the public OpenAPI surface at https://devs.adaptyvbio.com/api/v1 with
the things every real integration ends up needing: bearer auth, retry with
backoff that honours ``Retry-After``, cursor-free pagination over the list
endpoints, and typed access to the handful of response fields that callers
actually read.

The API uses Biscuit bearer tokens. Tokens encode org membership and
capabilities and can be attenuated via ``POST /tokens/attenuate`` — see
:meth:`FoundryClient.attenuate_token`, which is how you hand a narrower
token to an automated pipeline.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Mapping, Sequence

import requests

from .errors import APIError, AuthError, RateLimited

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://devs.adaptyvbio.com"
API_PREFIX = "/api/v1"

ExperimentType = Literal[
    "affinity",
    "screening",
    "thermostability",
    "fluorescence",
    "expression",
    "epitope_binning",
    "enzyme_activity",
]
Method = Literal["bli", "spr"]

#: Experiment types that the API requires a catalog ``target_id`` for.
TARGET_REQUIRED = {"affinity", "screening", "epitope_binning"}
#: Experiment types that require a ``method`` (and reject it otherwise).
METHOD_REQUIRED = {"affinity", "screening"}

RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class CostEstimate:
    """Result of ``POST /experiments/cost-estimate``.

    ``total_cents`` is ``None`` when Foundry returns an *incomplete* estimate,
    which happens for targets without self-service pricing. Callers must treat
    that as "unknown cost", never as "free" — :class:`~.guardrails.SpendGuard`
    refuses to approve an unpriced experiment for exactly this reason.
    """

    total_cents: int | None
    pricing_version: str | None
    warnings: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def total_usd(self) -> float | None:
        return None if self.total_cents is None else self.total_cents / 100

    @property
    def is_complete(self) -> bool:
        return self.total_cents is not None

    @classmethod
    def from_response(cls, body: Mapping[str, Any]) -> CostEstimate:
        breakdown = body.get("breakdown") or {}
        return cls(
            total_cents=breakdown.get("total_cents"),
            pricing_version=breakdown.get("pricing_version"),
            warnings=tuple(body.get("warnings") or ()),
            raw=body,
        )


def build_experiment_spec(
    *,
    experiment_type: ExperimentType,
    sequences: Mapping[str, str],
    target_id: str | None = None,
    method: Method | None = None,
    n_replicates: int | None = None,
    antigen_concentrations: Sequence[float] | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble an ``ExperimentSpec`` body, validating the type/field matrix.

    Foundry rejects a spec that sets a field its experiment type forbids, and
    reports every violation at once in a 400. Checking here instead turns a
    round trip into an immediate, local error message.
    """
    if not sequences:
        raise ValueError("sequences must not be empty")

    if experiment_type in TARGET_REQUIRED and not target_id:
        raise ValueError(f"{experiment_type} experiments require a catalog target_id")
    if experiment_type not in TARGET_REQUIRED and target_id:
        raise ValueError(f"{experiment_type} experiments must not set target_id")

    if experiment_type in METHOD_REQUIRED and not method:
        raise ValueError(f"{experiment_type} experiments require method='bli' or 'spr'")
    if experiment_type not in METHOD_REQUIRED and method:
        raise ValueError(f"{experiment_type} experiments must not set method")

    if n_replicates is not None and n_replicates < 1:
        raise ValueError("n_replicates must be >= 1")

    spec: dict[str, Any] = {
        "experiment_type": experiment_type,
        "sequences": dict(sequences),
    }
    if target_id:
        spec["target_id"] = target_id
    if method:
        spec["method"] = method
    if n_replicates is not None:
        spec["n_replicates"] = n_replicates
    if antigen_concentrations:
        spec["antigen_concentrations"] = list(antigen_concentrations)
    if parameters:
        spec["parameters"] = dict(parameters)
    return spec


class FoundryClient:
    """Thin, retrying HTTP client over the Foundry API.

    Every method maps to one documented endpoint. Read-only calls plus
    :meth:`cost_estimate` are non-billable, which is what makes a full
    dry-run rehearsal of a campaign free.
    """

    def __init__(
        self,
        token: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 5,
        session: requests.Session | None = None,
    ) -> None:
        token = token or os.environ.get("ADAPTYV_API_TOKEN")
        if not token:
            raise AuthError(
                "No API token. Pass token=... or set ADAPTYV_API_TOKEN. "
                "Tokens come from the Adaptyv Portal."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "User-Agent": "adaptyv-loop/0.1",
            }
        )

    # ---- transport -----------------------------------------------------

    def _request(
        self,
        verb: str,
        path: str,
        *,
        json_body: Any = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{API_PREFIX}{path}"
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._session.request(
                    verb, url, json=json_body, params=params, timeout=self.timeout
                )
            except requests.RequestException as exc:
                # Connection-level failures are transient often enough to retry.
                last_exc = exc
                if attempt == self.max_retries:
                    raise APIError(0, f"network failure after retries: {exc}") from exc
                self._sleep(attempt, None)
                continue

            if resp.status_code in (401, 403):
                raise AuthError(
                    f"HTTP {resp.status_code} on {verb} {path}. Token may be "
                    "revoked, expired, or lack the capability for this call."
                )

            if resp.status_code in RETRY_STATUSES and attempt < self.max_retries:
                self._sleep(attempt, resp.headers.get("Retry-After"))
                continue

            if resp.status_code == 429:
                raise RateLimited(f"still rate limited after {self.max_retries} retries")

            if not resp.ok:
                raise APIError(resp.status_code, self._error_message(resp), self._body(resp))

            return self._body(resp)

        raise APIError(0, f"exhausted retries on {verb} {path}: {last_exc}")

    def _sleep(self, attempt: int, retry_after: str | None) -> None:
        if retry_after:
            try:
                time.sleep(min(float(retry_after), 60.0))
                return
            except ValueError:
                pass
        # Exponential backoff with jitter, capped so a stuck job cannot wedge
        # a campaign for minutes at a time.
        delay = min(2.0**attempt, 30.0) * (0.5 + random.random() / 2)
        time.sleep(delay)

    @staticmethod
    def _body(resp: requests.Response) -> Any:
        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            return resp.text

    @staticmethod
    def _error_message(resp: requests.Response) -> str:
        body = FoundryClient._body(resp)
        if isinstance(body, Mapping):
            for key in ("message", "error", "detail", "title"):
                if key in body:
                    return str(body[key])
        return str(body)[:500] if body else resp.reason

    def _paginate(self, path: str, params: Mapping[str, Any] | None = None) -> Iterator[dict]:
        """Yield items from a list endpoint, following offset pagination.

        Foundry list endpoints return either a bare array or an object with an
        ``items``/``data`` array; both shapes are handled so a server-side
        change to the envelope does not break callers.
        """
        page_params = dict(params or {})
        limit = page_params.setdefault("limit", 100)
        offset = 0
        while True:
            page_params["offset"] = offset
            body = self._request("GET", path, params=page_params)
            if isinstance(body, Mapping):
                items = body.get("items") or body.get("data") or []
            else:
                items = body or []
            for item in items:
                yield item
            if len(items) < limit:
                return
            offset += len(items)

    # ---- identity & catalog -------------------------------------------

    def whoami(self) -> dict:
        """Identify the token and the orgs it can act for. Free, non-billable."""
        return self._request("GET", "/whoami")

    def list_targets(self, **params: Any) -> list[dict]:
        """List catalog target antigens. Free, non-billable."""
        return list(self._paginate("/targets", params))

    def get_target(self, target_id: str) -> dict:
        return self._request("GET", f"/targets/{target_id}")

    def find_target(self, query: str) -> list[dict]:
        """Case-insensitive search of the catalog by name or UniProt accession."""
        q = query.strip().lower()
        return [
            t
            for t in self.list_targets()
            if q in (t.get("name") or "").lower()
            or q == (t.get("uniprot_id") or "").lower()
        ]

    def attenuate_token(self, spec: Mapping[str, Any]) -> dict:
        """Mint a narrower token from the current one.

        Hand automated pipelines an attenuated token rather than the token that
        can also accept quotes. Attenuation is one-way and the derived token
        dies with its parent.
        """
        return self._request("POST", "/tokens/attenuate", json_body=spec)

    # ---- costing (free) ------------------------------------------------

    def cost_estimate(self, experiment_spec: Mapping[str, Any]) -> CostEstimate:
        """Price an experiment without creating it. Free, non-billable."""
        body = self._request(
            "POST",
            "/experiments/cost-estimate",
            json_body={"experiment_spec": dict(experiment_spec)},
        )
        return CostEstimate.from_response(body or {})

    # ---- experiments ---------------------------------------------------

    def create_experiment(
        self,
        name: str,
        experiment_spec: Mapping[str, Any],
        *,
        skip_draft: bool = False,
        auto_accept_quote: bool = False,
    ) -> dict:
        """Create an experiment.

        ``auto_accept_quote=True`` finalizes a Stripe quote, creates an invoice
        and advances the experiment without any human step. Do not set it from
        application code — route it through
        :meth:`~.guardrails.SpendGuard.approve` so the spend is bounded and
        logged.
        """
        payload: dict[str, Any] = {"name": name, "experiment_spec": dict(experiment_spec)}
        if skip_draft:
            payload["skip_draft"] = True
        if auto_accept_quote:
            payload["auto_accept_quote"] = True
        return self._request("POST", "/experiments", json_body=payload)

    def get_experiment(self, experiment_id: str) -> dict:
        return self._request("GET", f"/experiments/{experiment_id}")

    def list_experiments(self, **params: Any) -> list[dict]:
        return list(self._paginate("/experiments", params))

    def add_sequences(self, experiment_code: str, sequences: Sequence[Mapping[str, Any]]) -> dict:
        """Append sequences to a *draft* experiment, keyed by human-readable code.

        Only drafts accept new sequences; once confirmed you must create a new
        experiment instead.
        """
        return self._request(
            "POST",
            "/sequences",
            json_body={"experiment_code": experiment_code, "sequences": list(sequences)},
        )

    def submit_experiment(self, experiment_id: str) -> dict:
        return self._request("POST", f"/experiments/{experiment_id}/submit")

    def get_quote(self, experiment_id: str) -> dict:
        return self._request("GET", f"/experiments/{experiment_id}/quote")

    def confirm_quote(self, experiment_id: str) -> dict:
        """Accept the quote and create an invoice. **This commits real money.**"""
        return self._request("POST", f"/experiments/{experiment_id}/quote/confirm")

    def get_updates(self, experiment_id: str) -> list[dict]:
        return list(self._paginate(f"/experiments/{experiment_id}/updates"))

    def get_results(self, experiment_id: str) -> list[dict]:
        return list(self._paginate(f"/experiments/{experiment_id}/results"))

    def wait_for_results(
        self,
        experiment_id: str,
        *,
        poll_seconds: float = 900.0,
        timeout_seconds: float = 30 * 24 * 3600,
        on_status: Any = None,
    ) -> list[dict]:
        """Poll until the experiment reaches a terminal state, then return results.

        Assays take on the order of weeks, so the default cadence is 15 minutes
        and the default ceiling is 30 days. Terminal states are ``done`` and
        ``canceled``.
        """
        deadline = time.monotonic() + timeout_seconds
        last_status = None
        while True:
            exp = self.get_experiment(experiment_id)
            status = exp.get("status")
            if status != last_status:
                log.info("experiment %s -> %s", experiment_id, status)
                if on_status:
                    on_status(status, exp)
                last_status = status
            if status == "done":
                return self.get_results(experiment_id)
            if status == "canceled":
                raise APIError(0, f"experiment {experiment_id} was canceled")
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"experiment {experiment_id} still {status!r} after "
                    f"{timeout_seconds / 3600:.1f}h"
                )
            time.sleep(poll_seconds)
