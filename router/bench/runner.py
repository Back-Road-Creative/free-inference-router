"""Benchmark runner: send tasks to the local gateway, score, persist.

Dogfoods the router's own guard rails: router.quota.QuotaLedger refuses a
call BEFORE dispatch when the provider's registry caps would be exceeded
(a 429 must never be how a cap is discovered), and router.killswitch is
checked before every call. Results append to a JSONL the reporter reads;
completed (task_id, model, run_idx) triples are skipped on resume, so a
day-capped campaign continues where it stopped.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from router import load_providers
from router.bench.sandbox import SandboxUnavailableError
from router.bench.schema import Task
from router.bench.scoring import TOOL_CALL_SCORERS, score
from router.killswitch import engaged
from router.quota import QuotaLedger

GATEWAY = "http://127.0.0.1:8080/v1"
# The `local/` prefix routes straight to a loopback Ollama (the sealed local
# agent's model) instead of the gateway; the prefix is stripped for the wire name.
LOCAL_BASE = "http://127.0.0.1:11434/v1"
LOCAL_PREFIX = "local/"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})
# Error prefix for a row whose exec-scorer could not run docker. The reply
# was never judged, so the row is RETRY-class, not a fail.
SANDBOX_ERROR_PREFIX = "sandbox:"


def _resolve(model: str) -> tuple[str, str]:
    """(base_url, wire model name) for a bench model id.

    SAFETY: a `local/` model may only ever dial loopback — the local lane
    carries no key and must not be redirectable off-box by a mutated base URL.
    """
    if not model.startswith(LOCAL_PREFIX):
        return GATEWAY, model
    host = urllib.parse.urlsplit(LOCAL_BASE).hostname
    if host not in _LOOPBACK_HOSTS:
        raise ValueError(
            f"local base URL {LOCAL_BASE!r} is not loopback — refusing to dial"
        )
    return LOCAL_BASE, model[len(LOCAL_PREFIX) :]


@dataclass(frozen=True)
class RunResult:
    task_id: str
    role: str
    model: str
    run_idx: int
    passed: bool
    status: int
    latency_ms: int
    usage_total_tokens: int | None
    error: str | None = None


def _provider_of(model: str) -> str:
    provider = model.split("/", 1)[0]
    return {"gemini": "google"}.get(provider, provider)


def _caps_for(model: str) -> dict[str, Any]:
    provider = _provider_of(model)
    model_id = model.split("/", 1)[1]
    for e in load_providers():
        if e["provider"] == provider and e["model_id"] == model_id:
            if e["status"] != "admitted":
                raise PermissionError(f"{model} is not admitted")
            return e
    raise KeyError(f"{model} not in registry")


def _chat(model: str, task: Task, timeout_s: int) -> tuple[int, dict[str, Any]]:
    base_url, wire_model = _resolve(model)
    payload: dict[str, Any] = {
        "model": wire_model,
        "messages": [{"role": "user", "content": task.prompt}],
        "max_tokens": task.max_tokens,
    }
    if task.tools is not None:
        payload["tools"] = task.tools
        payload["tool_choice"] = "required"
    if base_url == LOCAL_BASE:
        # Ollama's OpenAI endpoint lets a thinking model (qwen3.5) spend the
        # whole max_tokens budget on `reasoning` and return content="" —
        # seen live 2026-09-01: 3/3 calls finish_reason=length, empty content.
        # `reasoning_effort: none` is the knob Ollama 0.32 honours.
        payload["reasoning_effort"] = "none"
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except ValueError:
            return e.code, {}


def _done_keys(out_path: Path) -> set[tuple[str, str, int]]:
    if not out_path.exists():
        return set()
    keys = set()
    for line in out_path.read_text().splitlines():
        try:
            r = json.loads(line)
            # Only a 200 completes a triple: a 429/5xx row is RETRY-class
            # (router/retry.py) and must be re-attempted on the next resume —
            # seen live 2026-08-31 when OpenRouter per-minute 429s burned 42
            # triples that would otherwise have been skipped forever. A
            # sandbox outage under an exec-scorer is RETRY-class too.
            unscored = str(r.get("error") or "").startswith(SANDBOX_ERROR_PREFIX)
            if r.get("status") == 200 and not unscored:
                keys.add((r["task_id"], r["model"], r["run_idx"]))
        except (ValueError, KeyError):
            continue
    return keys


def run_campaign(
    models: list[str],
    tasks: list[Task],
    out_path: Path,
    runs: int = 3,
    ledger: QuotaLedger | None = None,
) -> list[RunResult]:
    """Run every (task, model, run) not already in out_path. Quota-refused
    calls are SKIPPED silently (resume tomorrow); kill switch aborts."""
    ledger = ledger or QuotaLedger(out_path.with_suffix(".ledger.json"))
    done = _done_keys(out_path)
    results: list[RunResult] = []
    with out_path.open("a") as fh:
        for model in models:
            entry = _caps_for(model)
            provider = _provider_of(model)
            for task in tasks:
                for run_idx in range(runs):
                    if (task.task_id, model, run_idx) in done:
                        continue
                    if engaged():
                        raise RuntimeError("kill switch engaged — stopping")
                    est_tokens = len(task.prompt) // 3 + task.max_tokens
                    verdict = ledger.check(
                        provider, entry, est_tokens, datetime.now(timezone.utc)
                    )
                    if not verdict.allowed and verdict.reason in ("rpm", "tpm"):
                        time.sleep(61)  # minute bucket; try once more
                        verdict = ledger.check(
                            provider, entry, est_tokens, datetime.now(timezone.utc)
                        )
                    if not verdict.allowed:
                        continue  # day-capped (or still minute-capped): resume later
                    t0 = time.monotonic()
                    status, body = _chat(model, task, int(entry["timeout_s"]))
                    ms = int((time.monotonic() - t0) * 1000)
                    ledger.record(provider, est_tokens, datetime.now(timezone.utc))
                    passed, err = False, None
                    if status == 200:
                        msg = body.get("choices", [{}])[0].get("message", {})
                        if task.scorer in TOOL_CALL_SCORERS:
                            payload: Any = msg.get("tool_calls")
                        else:
                            content = msg.get("content") or ""
                            if isinstance(content, list):
                                # Gemini via the gateway can return content
                                # as a parts list (seen live: scorer crashed
                                # on list.find). Join the text parts.
                                content = "".join(
                                    p.get("text", "")
                                    for p in content
                                    if isinstance(p, dict)
                                )
                            payload = content
                        try:
                            passed = score(task.scorer, payload, task.scorer_args)
                        except SandboxUnavailableError as e:
                            # Docker could not run the reply: no verdict.
                            err = f"{SANDBOX_ERROR_PREFIX}{e}"
                        except Exception as e:  # scorer bug — surface, don't hide
                            err = f"scorer:{e!r}"
                    else:
                        err = str(body.get("error", {}).get("message", ""))[:200]
                    usage = (body.get("usage") or {}).get("total_tokens")
                    rec = RunResult(
                        task.task_id,
                        task.role,
                        model,
                        run_idx,
                        passed,
                        status,
                        ms,
                        usage,
                        err,
                    )
                    fh.write(json.dumps(rec.__dict__) + "\n")
                    fh.flush()
                    results.append(rec)
    return results
