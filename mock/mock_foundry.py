"""A local stand-in for the Adaptyv Foundry API, serving realistic data.

This is not a hand-waved mock. Its responses are shaped to Adaptyv's real
published OpenAPI schema, and the data behind them is real:

* Targets come from the EGFR antigen used in the public competition.
* Cost estimates use the real $99/protein screening price.
* Results replay the actual wet-lab binding outcomes for whatever sequences
  were submitted, matched by the sequence string.

Run it behind Prism in proxy mode (see ``mock/serve.sh``) and every request
and response is validated against Adaptyv's real contract on the way through,
so the loop is exercised against the genuine schema without any API access.

Endpoints implemented are exactly those the design-test-learn loop touches:
whoami, targets, cost-estimate, create/submit/get experiment, and results.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backtest"))

from run_backtest import ensure_data, load_round2  # noqa: E402

PRICE_PER_PROTEIN_CENTS = 9_900
EGFR_TARGET_ID = "019a03da-b87f-7e15-8b02-cef171c9871d"


def _load_truth() -> dict[str, bool]:
    """Map every competition sequence to its real binder / non-binder outcome."""
    data_dir = Path(__file__).resolve().parent.parent / "backtest" / "data"
    ensure_data(data_dir)
    df = load_round2(data_dir)
    return {str(row.sequence): bool(row.hit) for row in df.itertuples()}


class FoundryState:
    """In-memory experiment store. Deterministic ids, no wall clock.

    Outcomes come from two sources. A sequence that appears in the public
    competition gets its real measured result. A novel sequence — anything
    ``adaptyv_loop.design`` generated — has no measurement in existence, so it
    is routed to :class:`~adaptyv_loop.bench.SimulatedBench`, which is loaded
    only if a design library manifest is present.
    """

    def __init__(self, library: str | Path | None = None) -> None:
        self.truth = _load_truth()
        self.bench = None
        self.scores: dict[str, float] = {}
        if library:
            self._load_library(Path(library))
        self.experiments: dict[str, dict] = {}
        self._counter = 0

    def _load_library(self, path: Path) -> None:
        from adaptyv_loop.bench import SimulatedBench

        payload = json.loads(path.read_text())
        seqs = [d["sequence"] for d in payload["designs"]]
        scores = [d["score"] for d in payload["designs"]]
        self.scores = dict(zip(seqs, scores))
        self.bench = SimulatedBench.calibrate(seqs, scores)
        print(f"simulated bench calibrated on {len(seqs)} novel designs "
              f"(intercept {self.bench.intercept:.3f})")

    def outcome(self, sequence: str) -> str:
        """Real measurement where one exists, simulated where none can."""
        seq = str(sequence)
        if seq in self.truth:
            return "true" if self.truth[seq] else "false"
        if self.bench is not None and seq in self.scores:
            return "true" if self.bench.measure(seq, self.scores[seq]) else "false"
        return "unknown"

    def create(self, name: str, spec: dict) -> dict:
        """Returns CreateExpResponse — note the id is keyed `experiment_id`."""
        self._counter += 1
        exp_id = f"019b8da3-0000-0000-0000-{self._counter:012d}"
        self.experiments[exp_id] = {
            "id": exp_id,
            "code": f"DEMO-{self._counter:03d}",
            "name": name,
            "status": "in_queue",
            "experiment_type": spec.get("experiment_type"),
            "sequences": spec.get("sequences", {}),
            "spec_info": {
                "experiment_type": spec.get("experiment_type"),
                "method": spec.get("method"),
                "n_replicates": spec.get("n_replicates"),
            },
        }
        return {
            "experiment_id": exp_id,
            "error": None,
            "stripe_invoice_id": None,
            "stripe_hosted_invoice_url": None,
        }

    def submit(self, exp_id: str) -> dict:
        """Returns ExperimentConfirmationResponse."""
        exp = self.experiments[exp_id]
        previous = exp["status"]
        exp["status"] = "in_production"
        return {
            "experiment_id": exp_id,
            "previous_status": previous,
            "status": "in_production",
            "confirmed_at": "2026-07-22T00:00:00Z",
            "stripe_invoice_url": None,
        }

    def get(self, exp_id: str) -> dict:
        """Returns ExpInfo. First read after submission reports `done`.

        The mock models the state-machine endpoints, not the three-week wait.
        """
        exp = self.experiments[exp_id]
        exp["status"] = "done"
        return {
            "id": exp_id,
            "code": exp["code"],
            "name": exp["name"],
            "status": "done",
            "experiment_spec": exp["spec_info"],
            "created_at": "2026-07-22T00:00:00Z",
            "results_status": "all",
            "experiment_url": f"https://foundry.adaptyvbio.com/experiments/{exp_id}",
        }

    def results(self, exp_id: str) -> dict:
        """Returns a paginated list of ResultInfo, one AffinityResult per design."""
        exp = self.experiments[exp_id]
        summary = []
        for name, seq in exp["sequences"].items():
            binding = self.outcome(seq)
            strength = "strong" if binding == "true" else "none"
            summary.append({
                "result_type": "affinity",
                "sequence": {"aa_string": str(seq), "name": name, "control": False},
                "kd_units": "M",
                "binding": binding,
                "binding_strength": strength,
                "positive_control": False,
                "performance": {},
                "replicates": [],
            })
        return {
            "items": [{
                # Must be a real UUID: the spec declares format "uuid" and the
                # contract check rejects anything else.
                "id": f"019b8da3-0001-0000-0000-{exp_id[-12:]}",
                "title": f"{exp['name']} binding",
                "experiment_id": exp_id,
                "result_type": "affinity",
                "created_at": "2026-07-22T00:00:00Z",
                "summary": summary,
                "metadata": {},
                "data_package_url": None,
            }],
            "total": 1, "offset": 0, "count": 1,
        }


STATE: FoundryState | None = None


def cost_estimate(spec: dict) -> dict:
    seq_count = len(spec.get("sequences") or {})
    replicates = spec.get("n_replicates") or 1
    subtotal = seq_count * replicates * PRICE_PER_PROTEIN_CENTS
    return {
        "breakdown": {
            "pricing_version": "v1_2026-01-20",
            "assay": {
                "experiment_type": spec.get("experiment_type", "screening"),
                "sequence_count": seq_count,
                "n_replicates": replicates,
                "unit_price_cents": PRICE_PER_PROTEIN_CENTS,
                "replicate_price_cents": PRICE_PER_PROTEIN_CENTS,
                "subtotal_cents": subtotal,
            },
            "total_cents": subtotal,
        },
        "incomplete": None,
        "warnings": [],
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):  # keep the demo output clean
        pass

    def _send(self, code: int, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):  # noqa: N802
        assert STATE is not None
        path = urlparse(self.path).path
        if path == "/api/v1/whoami":
            return self._send(200, {
                "user_id": "019b8da3-1111-2222-3333-444455556666",
                "active_organization_id": "019b8da3-4a91-16c6-fa94-619212bee6a6",
                "organizations": [{
                    "id": "019b8da3-4a91-16c6-fa94-619212bee6a6",
                    "name": "Demo Org", "role": "member", "active": True,
                }],
                "permissions": ["experiment:read", "experiment:create"],
                "token_expires_at": "2026-12-31T23:59:59Z",
            })
        if path == "/api/v1/targets":
            return self._send(200, {
                "items": [{
                    "id": EGFR_TARGET_ID,
                    "name": "Human EGFR / ErbB1 (competition antigen)",
                    "vendor_name": "ACRO Biosystems",
                    "catalog_number": "EGF-H5222",
                    "uniprot_id": "P00533",
                    "url": f"https://targets.adaptyvbio.com/protein/{EGFR_TARGET_ID}",
                    # Matches the TargetPricing "per_sequence" branch of the real schema.
                    "pricing": {
                        "type": "per_sequence",
                        "price_per_sequence_cents": PRICE_PER_PROTEIN_CENTS,
                    },
                    "details": None,
                }],
                "total": 1, "offset": 0, "count": 1,
            })
        if path.startswith("/api/v1/experiments/") and path.endswith("/results"):
            exp_id = path.split("/")[4]
            return self._send(200, STATE.results(exp_id))
        if path.startswith("/api/v1/experiments/"):
            exp_id = path.split("/")[4]
            return self._send(200, STATE.get(exp_id))
        return self._send(404, {"message": f"no route for {path}"})

    def do_POST(self):  # noqa: N802
        assert STATE is not None
        path = urlparse(self.path).path
        body = self._read_json()
        if path == "/api/v1/experiments/cost-estimate":
            return self._send(200, cost_estimate(body.get("experiment_spec", {})))
        if path == "/api/v1/experiments":
            return self._send(201, STATE.create(body["name"], body["experiment_spec"]))
        if path.endswith("/submit"):
            return self._send(200, STATE.submit(path.split("/")[4]))
        return self._send(404, {"message": f"no route for {path}"})


def main() -> int:
    global STATE
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4011
    library = sys.argv[2] if len(sys.argv) > 2 else None
    if library and not Path(library).exists():
        library = None
    STATE = FoundryState(library)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"mock Foundry on http://127.0.0.1:{port} "
          f"({len(STATE.truth)} real EGFR outcomes loaded)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
