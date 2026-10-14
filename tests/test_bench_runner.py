"""Contract tests for router.bench.runner — no network, fake gateway."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest

import router
import router.bench.runner as runner_mod
from router.bench.runner import RunResult, run_campaign
from router.bench.sandbox import SandboxUnavailableError
from router.bench.schema import Task


def _task(**over: Any) -> Task:
    base: dict[str, Any] = {
        "task_id": "t1",
        "role": "triage",
        "prompt": "Reply with exactly the word: pong",
        "scorer": "exact_word",
        "scorer_args": {"word": "pong"},
        "max_tokens": 16,
    }
    base.update(over)
    return Task(**base)


GROQ = "groq/openai/gpt-oss-20b"


@pytest.fixture()
def fake_gateway(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def _chat(model: str, task: Task, timeout_s: int) -> tuple[int, dict[str, Any]]:
        calls.append({"model": model, "task": task.task_id})
        return 200, {
            "model": model.split("/", 1)[1],
            "choices": [{"message": {"content": "pong"}}],
            "usage": {"total_tokens": 7},
        }

    monkeypatch.setattr(runner_mod, "_chat", _chat)
    monkeypatch.setattr(runner_mod, "engaged", lambda: False)
    return calls


def test_campaign_runs_scores_and_persists(
    tmp_path: Path, fake_gateway: list[dict[str, Any]]
) -> None:
    out = tmp_path / "results.jsonl"
    res = run_campaign([GROQ], [_task()], out, runs=3)
    assert len(res) == 3 and all(r.passed for r in res)
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert {r["run_idx"] for r in lines} == {0, 1, 2}
    assert all(r["usage_total_tokens"] == 7 for r in lines)


def test_campaign_resume_skips_completed_triples(
    tmp_path: Path, fake_gateway: list[dict[str, Any]]
) -> None:
    out = tmp_path / "results.jsonl"
    run_campaign([GROQ], [_task()], out, runs=3)
    n_calls = len(fake_gateway)
    res2 = run_campaign([GROQ], [_task()], out, runs=3)
    assert res2 == [] and len(fake_gateway) == n_calls, "resume must not re-call"


def test_campaign_refuses_non_admitted_model(
    tmp_path: Path, fake_gateway: list[dict[str, Any]]
) -> None:
    with pytest.raises((PermissionError, KeyError)):
        run_campaign(["openrouter/not/a-model:free"], [_task()], tmp_path / "r.jsonl")


def test_campaign_kill_switch_aborts(
    tmp_path: Path, fake_gateway: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_mod, "engaged", lambda: True)
    with pytest.raises(RuntimeError, match="kill switch"):
        run_campaign([GROQ], [_task()], tmp_path / "r.jsonl")
    assert fake_gateway == []


def test_campaign_day_cap_skips_without_calling(
    tmp_path: Path, fake_gateway: list[dict[str, Any]]
) -> None:
    from datetime import datetime, timezone

    from router.quota import QuotaLedger

    ledger = QuotaLedger(tmp_path / "ledger.json")
    entry = runner_mod._caps_for(GROQ)
    for _ in range(int(entry["rpd"])):  # burn the whole day budget
        ledger.record("groq", 1, datetime.now(timezone.utc))
    res = run_campaign([GROQ], [_task()], tmp_path / "r.jsonl", runs=1, ledger=ledger)
    assert res == [] and fake_gateway == [], "capped call must never reach the gateway"


def test_failed_status_recorded_not_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_mod, "engaged", lambda: False)
    monkeypatch.setattr(
        runner_mod,
        "_chat",
        lambda model, task, timeout_s: (429, {"error": {"message": "rate limited"}}),
    )
    out = tmp_path / "r.jsonl"
    res = run_campaign([GROQ], [_task()], out, runs=1)
    assert len(res) == 1 and not res[0].passed and res[0].status == 429
    assert "rate limited" in (res[0].error or "")


def test_gemini_model_maps_to_google_registry_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert runner_mod._provider_of("gemini/gemini-3.5-flash") == "google"
    # Re-admitted 2026-09-02 with the AI Studio free-tier caps; the row is
    # billing-gated, so the runner builds caps from it only once the private
    # overlay (FIR_INPUTS_DIR/providers.local.json) attests it.
    (tmp_path / "providers.local.json").write_text(
        json.dumps(
            [
                {
                    "provider": "google",
                    "model_id": "gemini-3.5-flash",
                    "billing_disabled_attested": "2026-01-01 JP: fixture",
                },
            ]
        )
    )
    monkeypatch.setenv("FIR_INPUTS_DIR", str(tmp_path))
    caps = runner_mod._caps_for("gemini/gemini-3.5-flash")
    assert (caps["rpm"], caps["rpd"], caps["tpm"]) == (5, 20, 250000)


def test_runner_refuses_a_billing_gated_row_without_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FIR_INPUTS_DIR", raising=False)
    with pytest.raises(PermissionError, match="attest"):
        runner_mod._caps_for("gemini/gemini-3.5-flash")


def test_runresult_carries_no_prompt_or_reply_fields() -> None:
    fields = set(RunResult.__dataclass_fields__)
    assert not {"prompt", "reply", "content", "messages"} & fields


def test_resume_rerun_non_200_triples(
    tmp_path: Path, fake_gateway: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 429 row is RETRY-class, not done: the next campaign must re-call it."""
    out = tmp_path / "r.jsonl"
    monkeypatch.setattr(
        runner_mod,
        "_chat",
        lambda model, task, timeout_s: (429, {"error": {"message": "rate limited"}}),
    )
    run_campaign([GROQ], [_task()], out, runs=1)
    # second campaign with a healthy (fake) gateway must re-run the triple
    monkeypatch.setattr(
        runner_mod,
        "_chat",
        lambda model, task, timeout_s: (
            200,
            {
                "choices": [{"message": {"content": "pong"}}],
                "usage": {"total_tokens": 5},
            },
        ),
    )
    res2 = run_campaign([GROQ], [_task()], out, runs=1)
    assert len(res2) == 1 and res2[0].passed and res2[0].status == 200


def test_sandbox_outage_is_retried_not_kept_as_a_fail(
    tmp_path: Path, fake_gateway: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exec-scorer that cannot run docker has not judged the reply. The
    row says so (`sandbox:`) and is RETRY-class like a 429: the next
    campaign re-runs the triple instead of keeping a fail nobody scored."""
    out = tmp_path / "r.jsonl"

    def _outage(scorer: str, payload: Any, args: dict[str, Any]) -> bool:
        raise SandboxUnavailableError("docker stalled: no result within 155s")

    monkeypatch.setattr(runner_mod, "score", _outage)
    res = run_campaign([GROQ], [_task()], out, runs=1)
    assert len(res) == 1 and not res[0].passed and res[0].status == 200
    assert res[0].error == "sandbox:docker stalled: no result within 155s"

    monkeypatch.setattr(runner_mod, "score", lambda scorer, payload, args: True)
    res2 = run_campaign([GROQ], [_task()], out, runs=1)
    assert len(res2) == 1 and res2[0].passed
    assert len(fake_gateway) == 2


def test_scorer_bug_is_surfaced_and_not_retried(
    tmp_path: Path, fake_gateway: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any other scorer exception is a bug in the scorer, not an outage:
    surfaced in the row and done, so a resume does not re-call the model."""
    out = tmp_path / "r.jsonl"

    def _bug(scorer: str, payload: Any, args: dict[str, Any]) -> bool:
        raise KeyError("needles")

    monkeypatch.setattr(runner_mod, "score", _bug)
    res = run_campaign([GROQ], [_task()], out, runs=1)
    assert len(res) == 1 and not res[0].passed
    assert res[0].error == "scorer:KeyError('needles')"

    assert run_campaign([GROQ], [_task()], out, runs=1) == []
    assert len(fake_gateway) == 1


def test_list_form_content_is_joined_for_text_scorers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gemini via the gateway can return content as a parts list."""
    monkeypatch.setattr(runner_mod, "engaged", lambda: False)
    monkeypatch.setattr(
        runner_mod,
        "_chat",
        lambda model, task, timeout_s: (
            200,
            {
                "choices": [{"message": {"content": [{"text": "po"}, {"text": "ng"}]}}],
                "usage": {"total_tokens": 5},
            },
        ),
    )
    res = run_campaign([GROQ], [_task()], tmp_path / "r.jsonl", runs=1)
    assert len(res) == 1 and res[0].passed


# ------------------------------------------------------------------ local/
LOCAL = "local/qwen3.5:4b"


def test_local_model_resolves_to_loopback_ollama() -> None:
    base_url, wire_model = runner_mod._resolve(LOCAL)
    assert base_url == "http://127.0.0.1:11434/v1"
    assert wire_model == "qwen3.5:4b", "local/ prefix must be stripped on the wire"


def test_non_local_model_keeps_gateway() -> None:
    base_url, wire_model = runner_mod._resolve(GROQ)
    assert base_url == runner_mod.GATEWAY and wire_model == GROQ


def test_non_loopback_local_base_url_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SAFETY: a local/ model must never dial anything but loopback."""
    monkeypatch.setattr(runner_mod, "LOCAL_BASE", "http://evil.example.com:11434/v1")
    with pytest.raises(ValueError, match="loopback"):
        runner_mod._resolve(LOCAL)


def test_local_chat_sends_no_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _Resp:
        status = 200

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "pong"}}]}).encode()

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def _urlopen(req: Any, timeout: int | None = None) -> _Resp:
        captured["req"] = req
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    status, body = runner_mod._chat(LOCAL, _task(), 5)
    req = captured["req"]
    assert status == 200
    assert req.full_url == "http://127.0.0.1:11434/v1/chat/completions"
    assert not req.has_header("Authorization"), "local lane must carry no key"
    sent = json.loads(req.data)
    assert sent["model"] == "qwen3.5:4b"
    # a thinking model otherwise burns max_tokens on reasoning, content ""
    assert sent["reasoning_effort"] == "none"


def test_gateway_chat_does_not_force_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _Resp:
        status = 200

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "pong"}}]}).encode()

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def _urlopen(req: Any, timeout: int | None = None) -> _Resp:
        captured["req"] = req
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    runner_mod._chat("groq/llama", _task(), 5)
    assert "reasoning_effort" not in json.loads(captured["req"].data)


def test_local_registry_row_admitted_after_smoke() -> None:
    entry = next(
        e
        for e in router.load_providers()
        if e["provider"] == "local" and e["model_id"] == "qwen3.5:4b"
    )
    # 3/3 live smoke 2026-09-01 (tool_call_named, contains_all x2) via loopback
    assert entry["status"] == "admitted"
    assert entry["last_smoke"] == "2026-09-01T19:03:00Z"
    assert entry["data_policy_class"] == "no-training"
    assert set(entry["allowed_roles"]) == {"synthetic-benchmark", "public-repo-read"}
    assert entry["price_usd_per_mtok_in"] == 0
    assert entry["price_usd_per_mtok_out"] == 0
    assert all(entry[k] is None for k in ("rpm", "rpd", "tpm", "tpd"))
