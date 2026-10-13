#!/usr/bin/env bash
# Fetch the Laya ONNX weights for OFFLINE evaluation of router/decide.py
# (JP decision D1, 2026-09-22): onnxruntime and the Laya weights are
# approved for offline evaluation only -- this script never wires them
# into live selection, and router/decide.py itself is never imported by
# router.selector (tests/test_decide_not_wired.py asserts the boundary).
#
# Source: receptron/laya-onnx on huggingface.co -- an ONNX Runtime export
# of convaiinnovations/laya (421M-param non-autoregressive typed-decision
# model, Apache-2.0). Weights are Convai Innovations'; the export itself is
# MIT (https://github.com/receptron/laya). Verified 2026-09-23: the repo
# card confirms base_model=convaiinnovations/laya and the graph's real
# input/output names -- see README.md's "Getting the weights" section.
#
# Usage (run from this service's directory):
#   bash scripts/laya-fetch.sh
#
# Env overrides:
#   LAYA_REPO   HF repo id (default: receptron/laya-onnx)
#   LAYA_FILES  space-separated file paths inside the repo to fetch
#   LAYA_DIR    local destination directory
#               (default: ${XDG_CACHE_HOME:-$HOME/.cache}/laya)
#   LAYA_MODEL_PATH  path to the .onnx file the probe step opens
#                    (default: $LAYA_DIR/laya.onnx)
#
# Idempotent: a file already on disk whose sha256 matches the last
# recorded fetch for that path is not re-downloaded. Only huggingface.co
# is ever contacted.

set -euo pipefail

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SERVICE_DIR"

LAYA_REPO="${LAYA_REPO:-receptron/laya-onnx}"
LAYA_FILES="${LAYA_FILES:-laya.onnx laya.onnx.data laya_config.json tokenizer/tokenizer.json tokenizer/tokenizer_config.json}"
LAYA_DIR="${LAYA_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/laya}"
LAYA_MODEL_PATH="${LAYA_MODEL_PATH:-$LAYA_DIR/laya.onnx}"

mkdir -p "$LAYA_DIR"
MANIFEST="$LAYA_DIR/MANIFEST.txt"
touch "$MANIFEST"

if [ ! -d "$SERVICE_DIR/.venv" ]; then
  echo "laya-fetch: creating .venv" >&2
  python3 -m venv "$SERVICE_DIR/.venv"
fi
echo "laya-fetch: installing the 'laya' extra (onnxruntime) into .venv" >&2
"$SERVICE_DIR/.venv/bin/pip" install -q -e ".[laya]"

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

# A manifest line already recording this (relpath, sha256) pair means the
# file on disk right now is exactly what we last fetched -- skip it.
already_fetched() {
  local relpath="$1" sha="$2"
  /bin/grep -qF "$(printf '%s\t%s\t' "$relpath" "$sha")" "$MANIFEST" 2>/dev/null
}

for rel in $LAYA_FILES; do
  dest="$LAYA_DIR/$rel"
  mkdir -p "$(dirname "$dest")"
  url="https://huggingface.co/$LAYA_REPO/resolve/main/$rel"

  if [ -f "$dest" ]; then
    existing_sha="$(sha256_of "$dest")"
    if already_fetched "$rel" "$existing_sha"; then
      echo "laya-fetch: $rel already present, sha256 matches manifest -- skipping" >&2
      continue
    fi
  fi

  echo "laya-fetch: downloading $rel from $url" >&2
  tmp="$dest.part"
  if ! curl -fL --retry 3 -o "$tmp" "$url"; then
    echo "laya-fetch: FAILED to download $rel from $url" >&2
    rm -f "$tmp"
    exit 1
  fi
  mv "$tmp" "$dest"
  new_sha="$(sha256_of "$dest")"
  fetched_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\n' "$rel" "$new_sha" "$url" "$fetched_at" >> "$MANIFEST"
  echo "laya-fetch: fetched $rel sha256=$new_sha" >&2
done

echo "laya-fetch: probing ONNX graph at $LAYA_MODEL_PATH" >&2
if ! "$SERVICE_DIR/.venv/bin/python" - "$LAYA_MODEL_PATH" <<'PY'
import sys

import onnxruntime as ort

model_path = sys.argv[1]
try:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
except Exception as exc:  # noqa: BLE001 -- surface any onnxruntime error clearly
    print(f"laya-fetch: FAILED to open ONNX model {model_path}: {exc}", file=sys.stderr)
    sys.exit(1)

print("== inputs ==")
for i in sess.get_inputs():
    print(f"{i.name}\tshape={i.shape}\ttype={i.type}")
print("== outputs ==")
for o in sess.get_outputs():
    print(f"{o.name}\tshape={o.shape}\ttype={o.type}")
PY
then
  echo "laya-fetch: probe failed" >&2
  exit 1
fi

echo "laya-fetch: done. Weights are in $LAYA_DIR (gitignored, offline eval only)." >&2
