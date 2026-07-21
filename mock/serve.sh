#!/usr/bin/env bash
# Bring up the contract-validated mock Foundry:
#
#   client -> Prism (:4010, validates against Adaptyv's real OpenAPI spec) -> mock (:4011, real EGFR data)
#
# Every request and response crossing Prism is checked against the genuine
# published contract, so demo_mock.py exercises the real schema with no API
# access. Ctrl-C tears both down.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-./.venv/bin/python}"
SPEC="backtest/foundry_openapi.json"

# --library also loads the generated design library, so novel sequences get an
# outcome from the calibrated simulated bench instead of "unknown".
LIBRARY=""
if [ "${1:-}" = "--library" ]; then
  LIBRARY="design_library.json"
  [ -f "$LIBRARY" ] || { echo "no $LIBRARY — run: python demo_end_to_end.py --build-library"; exit 1; }
fi

echo "starting mock Foundry (:4011) ..."
"$PYTHON" mock/mock_foundry.py 4011 $LIBRARY &
MOCK_PID=$!

cleanup() { kill "$MOCK_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

sleep 2
echo "starting Prism validating proxy (:4010 -> :4011) against Adaptyv's real spec ..."
echo "point the client at http://127.0.0.1:4010"
npx -y @stoplight/prism-cli@5 proxy "$SPEC" http://127.0.0.1:4011 -p 4010 --errors
