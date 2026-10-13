#!/usr/bin/env bash
# Opt-in LIVE network canary for the Phase 1 egress boundary. Never run by
# CI or by the router's tests/test_egress_compose.py — this touches the real
# network. Proves: a request to a non-admitted host through the gateway's
# proxy fails, and a request to an admitted host reaches the network.
set -euo pipefail

if [[ "${ROUTER_LIVE_CANARY:-}" != "1" ]]; then
  echo "refusing to run: set ROUTER_LIVE_CANARY=1 to run the live egress canary" >&2
  exit 1
fi

COMPOSE_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/docker-compose.yml"

cleanup() {
  docker compose -f "$COMPOSE_FILE" down >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker compose -f "$COMPOSE_FILE" up -d proxy

echo "-- non-admitted host (example.com) must FAIL --"
if docker compose -f "$COMPOSE_FILE" exec -T proxy sh -c \
    'wget -q -O- --timeout=5 -e use_proxy=yes -e https_proxy=http://127.0.0.1:8888 https://example.com'; then
  echo "FAIL: non-admitted host was reachable through the proxy" >&2
  exit 1
fi
echo "ok: example.com blocked"

echo "-- admitted host (api.groq.com) must be reachable --"
docker compose -f "$COMPOSE_FILE" exec -T proxy sh -c \
  'wget -q -O- --timeout=5 -e use_proxy=yes -e https_proxy=http://127.0.0.1:8888 https://api.groq.com/ >/dev/null'
echo "ok: api.groq.com reachable through the proxy"

echo "live canary passed"
