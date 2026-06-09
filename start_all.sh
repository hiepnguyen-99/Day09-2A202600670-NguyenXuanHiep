#!/usr/bin/env bash
set -euo pipefail

# Start all Legal Multi-Agent System services
# Registry must be first, then leaf agents, then orchestrators.

run_py() {
  if command -v uv >/dev/null 2>&1; then
    uv run python "$@"
  else
    python "$@"
  fi
}

pids=()

cleanup() {
  echo ""
  echo "Stopping services..."
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
  wait || true
  echo "All services stopped."
}

trap cleanup INT TERM EXIT

echo "Starting Registry service on port 10000..."
run_py -m registry &
pids+=($!)
sleep 2

echo "Starting Tax Agent on port 10102..."
run_py -m tax_agent &
pids+=($!)

echo "Starting Compliance Agent on port 10103..."
run_py -m compliance_agent &
pids+=($!)
sleep 3

echo "Starting Law Agent on port 10101..."
run_py -m law_agent &
pids+=($!)
sleep 3

echo "Starting Customer Agent on port 10100..."
run_py -m customer_agent &
pids+=($!)

echo ""
echo "All services started:"
echo "  Registry:          http://localhost:10000"
echo "  Customer Agent:    http://localhost:10100"
echo "  Law Agent:         http://localhost:10101"
echo "  Tax Agent:         http://localhost:10102"
echo "  Compliance Agent:  http://localhost:10103"
echo ""
echo "Run test_client.py to send a query:"
echo "  uv run python test_client.py"
echo ""
echo "Press Ctrl+C to stop all services."

# Wait forever until Ctrl+C
wait