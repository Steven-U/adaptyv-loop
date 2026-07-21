"""Client tests: spec validation, retry behaviour, and result parsing."""

from __future__ import annotations

import pytest
import requests

from adaptyv_loop.campaign import outcomes_from_results
from adaptyv_loop.client import CostEstimate, FoundryClient, build_experiment_spec
from adaptyv_loop.errors import APIError, AuthError

SEQS = {"a": "MKTVR", "b": "MKVWR"}


# ---- spec validation ---------------------------------------------------


def test_screening_spec_round_trips():
    spec = build_experiment_spec(
        experiment_type="screening", sequences=SEQS, target_id="t-1", method="bli"
    )
    assert spec["experiment_type"] == "screening"
    assert spec["target_id"] == "t-1"
    assert spec["method"] == "bli"


@pytest.mark.parametrize("experiment_type", ["affinity", "screening", "epitope_binning"])
def test_target_required_types_reject_a_missing_target(experiment_type):
    with pytest.raises(ValueError, match="require a catalog target_id"):
        build_experiment_spec(
            experiment_type=experiment_type, sequences=SEQS, method="bli"
        )


def test_thermostability_rejects_a_target():
    """The API rejects fields a type forbids; catching it locally saves a round trip."""
    with pytest.raises(ValueError, match="must not set target_id"):
        build_experiment_spec(
            experiment_type="thermostability", sequences=SEQS, target_id="t-1"
        )


def test_thermostability_rejects_a_method():
    with pytest.raises(ValueError, match="must not set method"):
        build_experiment_spec(
            experiment_type="thermostability", sequences=SEQS, method="bli"
        )


def test_affinity_requires_a_method():
    with pytest.raises(ValueError, match="require method"):
        build_experiment_spec(
            experiment_type="affinity", sequences=SEQS, target_id="t-1"
        )


def test_empty_sequences_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        build_experiment_spec(
            experiment_type="expression", sequences={}
        )


# ---- cost estimates ----------------------------------------------------


def test_complete_estimate_parses_cents():
    est = CostEstimate.from_response(
        {"breakdown": {"total_cents": 495_000, "pricing_version": "v1"}, "warnings": []}
    )
    assert est.is_complete and est.total_usd == pytest.approx(4_950.0)


def test_incomplete_estimate_is_not_zero():
    est = CostEstimate.from_response(
        {"breakdown": None, "incomplete": {}, "warnings": ["target lacks pricing"]}
    )
    assert not est.is_complete
    assert est.total_usd is None
    assert est.warnings == ("target lacks pricing",)


# ---- transport ---------------------------------------------------------


class FakeResponse:
    def __init__(self, status: int, payload=None, headers=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.reason = "fake"
        self.content = b"{}"

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = []

    def request(self, verb, url, **kwargs):
        self.calls.append((verb, url, kwargs))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def client_with(responses, **kwargs):
    return FoundryClient(
        token="t", session=FakeSession(responses), max_retries=3, **kwargs
    )


def test_missing_token_is_an_auth_error(monkeypatch):
    monkeypatch.delenv("ADAPTYV_API_TOKEN", raising=False)
    with pytest.raises(AuthError, match="No API token"):
        FoundryClient()


def test_token_goes_in_the_authorization_header():
    c = FoundryClient(token="secret", session=FakeSession([]))
    assert c._session.headers["Authorization"] == "Bearer secret"


def test_401_is_an_auth_error_and_is_not_retried():
    c = client_with([FakeResponse(401)])
    with pytest.raises(AuthError):
        c.whoami()
    assert len(c._session.calls) == 1


def test_server_error_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    c = client_with([FakeResponse(503), FakeResponse(500), FakeResponse(200, {"id": "u"})])
    assert c.whoami() == {"id": "u"}
    assert len(c._session.calls) == 3


def test_retry_after_header_is_honoured(monkeypatch):
    slept = []
    monkeypatch.setattr("time.sleep", slept.append)
    c = client_with([FakeResponse(429, headers={"Retry-After": "2"}), FakeResponse(200, {})])
    c.whoami()
    assert slept == [2.0]


def test_network_errors_are_retried(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    c = client_with([requests.ConnectionError("reset"), FakeResponse(200, {"ok": True})])
    assert c.whoami() == {"ok": True}


def test_client_error_is_surfaced_with_the_api_message():
    c = client_with([FakeResponse(400, {"message": "sequences: invalid residue X"})])
    with pytest.raises(APIError, match="invalid residue X") as exc:
        c.whoami()
    assert exc.value.status == 400


def test_pagination_follows_pages(monkeypatch):
    page1 = FakeResponse(200, {"items": [{"id": i} for i in range(100)]})
    page2 = FakeResponse(200, {"items": [{"id": 100}]})
    c = client_with([page1, page2])
    assert len(c.list_targets()) == 101


def test_pagination_accepts_a_bare_array():
    c = client_with([FakeResponse(200, [{"id": 1}])])
    assert c.list_targets() == [{"id": 1}]


def test_find_target_matches_name_or_uniprot():
    body = [
        {"id": "1", "name": "Human EGFR", "uniprot_id": "P00533"},
        {"id": "2", "name": "PD-L1", "uniprot_id": "Q9NZQ7"},
    ]
    assert FoundryClient(token="t", session=FakeSession([FakeResponse(200, body)])).find_target(
        "egfr"
    )[0]["id"] == "1"
    assert FoundryClient(token="t", session=FakeSession([FakeResponse(200, body)])).find_target(
        "Q9NZQ7"
    )[0]["id"] == "2"


# ---- result parsing ----------------------------------------------------


def test_outcomes_read_the_binding_call():
    results = [
        {
            "summary": [
                {"sequence": {"name": "d1"}, "binding": "true"},
                {"sequence": {"name": "d2"}, "binding": "false"},
            ]
        }
    ]
    assert outcomes_from_results(results) == {"d1": True, "d2": False}


def test_unknown_binding_is_dropped_not_counted_as_a_miss():
    """A construct that failed to express is missing data, not a failed binder.

    Counting it as a miss would slowly poison the method posteriors.
    """
    results = [{"summary": [{"sequence": {"name": "d3"}, "binding": "unknown"}]}]
    assert outcomes_from_results(results) == {}


def test_falls_back_to_binding_strength():
    results = [
        {
            "summary": [
                {"sequence": {"name": "d4"}, "binding_strength": "medium"},
                {"sequence": {"name": "d5"}, "binding_strength": "none"},
            ]
        }
    ]
    assert outcomes_from_results(results) == {"d4": True, "d5": False}


def test_entries_without_a_name_are_skipped():
    assert outcomes_from_results([{"summary": [{"binding": "true"}]}]) == {}


def test_empty_results_are_handled():
    assert outcomes_from_results([]) == {}
    assert outcomes_from_results([{"summary": None}]) == {}


# ---- response-shape quirks caught by contract-checking against the real spec ----


def test_campaign_reads_the_id_from_create_response(tmp_path):
    """CreateExpResponse keys the id as `experiment_id`; ExpInfo uses `id`.

    Reading only `id` silently yields None, and the campaign then submits and
    polls against a null experiment. Caught by running the client against
    Adaptyv's published OpenAPI spec behind a Prism validating proxy.
    """
    from adaptyv_loop import Campaign, Candidate, SpendPolicy, select_designs

    class Stub:
        def __init__(self, create_body):
            self.create_body = create_body
            self.submitted = []

        def cost_estimate(self, spec):
            return CostEstimate(total_cents=10_000, pricing_version="v1")

        def create_experiment(self, name, spec, *, auto_accept_quote=False):
            return self.create_body

        def submit_experiment(self, experiment_id):
            self.submitted.append(experiment_id)
            return {}

    for body, expected in (
        ({"experiment_id": "exp-42", "error": None}, "exp-42"),   # CreateExpResponse
        ({"id": "exp-7"}, "exp-7"),                               # ExpInfo-style
    ):
        stub = Stub(body)
        campaign = Campaign(
            stub,
            campaign_id=f"c-{expected}",
            target_id="t-1",
            policy=SpendPolicy(max_experiment_usd=1_000, max_campaign_usd=1_000),
            state_dir=tmp_path,
        )
        selection = select_designs(
            [Candidate(id="d1", sequence="MKT", method="m")], 1
        )
        record = campaign.submit_round(selection)
        assert record.experiment_id == expected
        assert stub.submitted == [expected]


def test_campaign_rejects_a_create_response_with_no_id(tmp_path):
    from adaptyv_loop import Campaign, Candidate, SpendPolicy, select_designs

    class Stub:
        def cost_estimate(self, spec):
            return CostEstimate(total_cents=10_000, pricing_version="v1")

        def create_experiment(self, name, spec, *, auto_accept_quote=False):
            return {"error": "something went wrong"}

    campaign = Campaign(
        Stub(),
        campaign_id="c-bad",
        target_id="t-1",
        policy=SpendPolicy(max_experiment_usd=1_000, max_campaign_usd=1_000),
        state_dir=tmp_path,
    )
    selection = select_designs([Candidate(id="d1", sequence="MKT", method="m")], 1)
    with pytest.raises(APIError, match="no id"):
        campaign.submit_round(selection)


def test_results_are_paginated_not_a_bare_array():
    """GET /experiments/{id}/results returns the standard list envelope."""
    envelope = {"items": [{"id": "r1", "summary": []}], "total": 1, "offset": 0, "count": 1}
    c = client_with([FakeResponse(200, envelope)])
    assert c.get_results("exp-1") == [{"id": "r1", "summary": []}]
