#!/usr/bin/env python3
"""Groq admission smoke — run against the LOCAL gateway (127.0.0.1:8080).

Needs no key on the client side: the gateway holds the credential. Uses
synthetic text only. Checks (plan Phase 2): model listed with exact id,
non-stream chat, streaming, tool-call argument shape, structured output,
returned model identity, usage accounting present. Prints one PASS/FAIL
line per check and a machine-readable JSON summary; exit 0 only if all
required checks pass.
"""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8080/v1"
# Default: the groq pair. Pass explicit gateway model ids as argv to smoke
# another provider, e.g.:
#   smoke_groq.py openrouter/thinkingmachines/inkling:free \
#                 openrouter/thinkingmachines/inkling-small:free
MODELS = sys.argv[1:] or ["groq/openai/gpt-oss-120b", "groq/openai/gpt-oss-20b"]
SYNTH = "Reply with exactly the word: pong"
results: dict[str, dict[str, object]] = {}


def call(
    path: str, payload: dict[str, object] | None = None, timeout: int = 60
) -> tuple[int, bytes]:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode() if payload else None,
        headers={"Content-Type": "application/json"},
        method="POST" if payload else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:  # noqa: PERF203
        return e.code, e.read()


def check(model: str) -> dict[str, object]:
    out: dict[str, object] = {}
    status, body = call("/models")
    listed = [m.get("id") for m in json.loads(body).get("data", [])]
    out["model_listed"] = status == 200 and model in listed
    # non-stream + identity + usage
    status, body = call(
        "/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": SYNTH}],
            "max_tokens": 16,
        },
    )
    ok = status == 200
    returned_model, usage = None, None
    if ok:
        j = json.loads(body)
        returned_model = j.get("model")
        usage = j.get("usage") or {}
    out["chat_200"] = ok
    out["identity_match"] = bool(returned_model) and str(returned_model).endswith(
        model.split("/", 1)[1]
    )
    out["returned_model"] = returned_model
    out["usage_accounted"] = bool(usage and usage.get("total_tokens"))
    # streaming
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": SYNTH}],
                "max_tokens": 16,
                "stream": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            first = r.readline().decode(errors="replace")
        out["stream_sse"] = first.startswith("data:")
    except Exception as e:  # noqa: BLE001
        out["stream_sse"] = False
        out["stream_error"] = repr(e)
    # tool-call argument shape
    status, body = call(
        "/chat/completions",
        {
            "model": model,
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Paris? Use the tool.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        },
    )
    tool_ok = False
    if status == 200:
        try:
            tc = json.loads(body)["choices"][0]["message"]["tool_calls"][0]
            args = json.loads(tc["function"]["arguments"])
            tool_ok = tc["function"]["name"] == "get_weather" and isinstance(
                args.get("city"), str
            )
        except Exception:  # noqa: BLE001
            tool_ok = False
    out["tool_call_shape"] = tool_ok
    # structured output
    status, body = call(
        "/chat/completions",
        {
            "model": model,
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": "Give a JSON object with key answer set to 42.",
                }
            ],
            "response_format": {"type": "json_object"},
        },
    )
    so_ok = False
    if status == 200:
        try:
            content = json.loads(body)["choices"][0]["message"]["content"]
            so_ok = isinstance(json.loads(content), dict)
        except Exception:  # noqa: BLE001
            so_ok = False
    out["structured_output"] = so_ok
    return out


def main() -> int:
    required = (
        "model_listed",
        "chat_200",
        "identity_match",
        "usage_accounted",
        "stream_sse",
        "tool_call_shape",
        "structured_output",
    )
    ok = True
    for model in MODELS:
        r = check(model)
        results[model] = r
        for k in required:
            good = bool(r.get(k))
            ok = ok and good
            extra = r.get("returned_model", "") if k == "identity_match" else ""
            print(f"{'PASS' if good else 'FAIL'}  {model}  {k}  {extra}")
    print(json.dumps(results, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
