#!/usr/bin/env bash
# up-with-keys.sh — start/stop the router gateway WITH provider keys.
# JP-only (self-sudos: keys come from the encrypted store via sops-get).
# Keys go straight from the store into the container environment; they are
# never echoed, never written to disk, never exported to the calling shell.
# Usage (from the service directory):  bash compose/up-with-keys.sh up
#                                      bash compose/up-with-keys.sh down
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
CMD="${1:-up}"
if [ "$(id -u)" -ne 0 ]; then exec sudo bash "$(readlink -f "$0")" "$@"; fi
case "$CMD" in
  down) docker compose -f "$HERE/docker-compose.yml" down; exit $? ;;
  up) ;;
  *) echo "usage: $0 up|down" >&2; exit 2 ;;
esac
for K in GROQ_API_KEY OPENROUTER_API_KEY GEMINI_API_KEY; do
  V=$(/usr/local/bin/sops-get "$K") || { echo "$K missing from store" >&2; exit 3; }
  export "$K=$V"
done
unset V
docker compose -f "$HERE/docker-compose.yml" up -d
unset GROQ_API_KEY OPENROUTER_API_KEY GEMINI_API_KEY
echo "stack up — gateway on 127.0.0.1:8080 (keys live only inside the container)"
