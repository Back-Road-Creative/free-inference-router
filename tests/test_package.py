"""The package loads its own tracked policy data and refuses malformed data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import router


def test_providers_load_and_admitted_set_matches_the_bars() -> None:
    entries = router.load_providers()
    assert entries, "registry must not be empty"
    admitted = {
        (e["provider"], e["model_id"]) for e in entries if e["status"] == "admitted"
    }
    assert admitted == {
        ("groq", "openai/gpt-oss-120b"),
        ("groq", "openai/gpt-oss-20b"),
        ("openrouter", "inclusionai/ling-3.0-flash-fin:free"),
        ("google", "gemini-3.5-flash"),
        ("google", "gemini-3.5-flash-lite"),
        ("local", "qwen3.5:4b"),
    }


def test_classification_is_default_deny() -> None:
    assert router.load_classification()["default"] == "deny"


def test_classification_refuses_non_deny(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "classification.json").write_text(json.dumps({"default": "allow"}))
    monkeypatch.setattr(router, "DATA_DIR", tmp_path)
    with pytest.raises(ValueError):
        router.load_classification()
