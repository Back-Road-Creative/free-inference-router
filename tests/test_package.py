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


# ------------------------------------------------------- private overlay

GOOGLE = ("google", "gemini-3.5-flash")


def _overlay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: object) -> None:
    (tmp_path / "providers.local.json").write_text(json.dumps(rows))
    monkeypatch.setenv("FIR_INPUTS_DIR", str(tmp_path))


def test_tracked_registry_carries_no_attestation_value() -> None:
    """The attestation names JP's cloud project; it is private data and never
    sits in the tracked (public) registry."""
    entries = json.loads((Path(router.DATA_DIR) / "providers.json").read_text())
    assert all(e.get("billing_disabled_attested") is None for e in entries)


def test_without_overlay_gated_rows_stay_unattested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FIR_INPUTS_DIR", raising=False)
    gated = [e for e in router.load_providers() if e.get("billing_gated")]
    assert gated and all(e["billing_disabled_attested"] is None for e in gated)


def test_missing_overlay_file_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FIR_INPUTS_DIR", str(tmp_path))
    gated = [e for e in router.load_providers() if e.get("billing_gated")]
    assert all(e["billing_disabled_attested"] is None for e in gated)


def test_overlay_attests_a_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _overlay(
        tmp_path,
        monkeypatch,
        [
            {
                "provider": GOOGLE[0],
                "model_id": GOOGLE[1],
                "billing_disabled_attested": "2026-01-01 JP: fixture",
            },
        ],
    )
    by_key = {(e["provider"], e["model_id"]): e for e in router.load_providers()}
    assert by_key[GOOGLE]["billing_disabled_attested"] == "2026-01-01 JP: fixture"
    other = by_key[("google", "gemini-3.5-flash-lite")]
    assert other["billing_disabled_attested"] is None, "overlay is per row"


def test_overlay_may_only_attest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The private file attests; it never admits, re-caps or re-classes a row."""
    _overlay(
        tmp_path,
        monkeypatch,
        [
            {"provider": GOOGLE[0], "model_id": GOOGLE[1], "status": "admitted"},
        ],
    )
    with pytest.raises(ValueError, match="status"):
        router.load_providers()


def test_overlay_row_must_name_a_registry_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _overlay(
        tmp_path,
        monkeypatch,
        [
            {
                "provider": "nobody",
                "model_id": "x",
                "billing_disabled_attested": "2026-01-01 JP: f",
            },
        ],
    )
    with pytest.raises(ValueError, match="nobody/x"):
        router.load_providers()


def test_overlay_must_be_a_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _overlay(tmp_path, monkeypatch, {"provider": "google"})
    with pytest.raises(ValueError, match="list"):
        router.load_providers()
