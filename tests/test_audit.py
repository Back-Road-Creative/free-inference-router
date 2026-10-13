from __future__ import annotations

import json
from pathlib import Path

import pytest

from router import audit as audit_mod


def test_unknown_key_raises(tmp_path: Path) -> None:
    log = audit_mod.AuditLog(tmp_path / "log.jsonl")
    with pytest.raises(ValueError):
        log.append(attempt_id="a1", provider="groq", prompt="secret prompt")


def test_key_shaped_value_raises_and_file_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    log = audit_mod.AuditLog(path)
    key = "gsk_" + "d" * 20
    with pytest.raises(ValueError):
        log.append(attempt_id="a1", provider="groq", reason=f"leaked {key}")
    assert not path.exists()


def test_key_shaped_value_after_valid_record_leaves_file_unchanged(
    tmp_path: Path,
) -> None:
    path = tmp_path / "log.jsonl"
    log = audit_mod.AuditLog(path)
    log.append(attempt_id="a1", provider="groq", status="ok")
    before = path.read_text()
    key = "AIzaSy" + "e" * 33
    with pytest.raises(ValueError):
        log.append(attempt_id="a2", provider="google", reason=f"leaked {key}")
    assert path.read_text() == before


def test_valid_record_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    log = audit_mod.AuditLog(path)
    log.append(
        attempt_id="a1",
        provider="groq",
        requested_model="m1",
        returned_model="m1",
        status="ok",
        latency_ms=120,
        tokens_in=10,
        tokens_out=20,
        lane="R1",
        reason="ok",
        at="2026-08-30T00:00:00Z",
    )
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["provider"] == "groq"
    assert record["returned_model"] == "m1"
    assert record["status"] == "ok"


def test_no_forbidden_field_names_on_disk(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    log = audit_mod.AuditLog(path)
    log.append(attempt_id="a1", provider="groq", status="ok")
    line = path.read_text()
    for forbidden in ("prompt", "messages", "content", "input", "output"):
        assert f'"{forbidden}"' not in line


def test_provider_and_returned_model_recorded(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    log = audit_mod.AuditLog(path)
    log.append(provider="openrouter", returned_model="m2")
    record = json.loads(path.read_text().splitlines()[0])
    assert record["provider"] == "openrouter"
    assert record["returned_model"] == "m2"
