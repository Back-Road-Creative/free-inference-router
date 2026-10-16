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
    """``FIR_INPUTS_DIR=""`` is the explicit off switch (the suite's default)."""
    monkeypatch.setenv("FIR_INPUTS_DIR", "")
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


def test_unset_env_finds_the_private_sibling_of_the_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No launcher exports the variable, so an unset env searches upward from
    the package for ``_private/free-inference-router/providers.local.json``;
    nothing found means no overlay."""
    pkg = tmp_path / "root" / "checkout" / "router"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(router, "DATA_DIR", pkg)
    monkeypatch.delenv("FIR_INPUTS_DIR", raising=False)
    assert router.private_overlay_path() is None
    private = tmp_path / "root" / "_private" / "free-inference-router"
    private.mkdir(parents=True)
    assert router.private_overlay_path() == private / "providers.local.json"
    monkeypatch.setenv("FIR_INPUTS_DIR", "")
    assert router.private_overlay_path() is None


@pytest.mark.parametrize("checkout", ["free-inference-router", "fir-clone", "x"])
def test_lookup_does_not_depend_on_the_checkout_folder_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkout: str
) -> None:
    """A clone finds ``_private/free-inference-router`` whatever its own folder
    is called: the lookup uses the fixed ``PRIVATE_DIR_NAME``."""
    pkg = tmp_path / "ws" / ".data" / "pilot" / checkout / "router"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(router, "DATA_DIR", pkg)
    monkeypatch.delenv("FIR_INPUTS_DIR", raising=False)
    private = tmp_path / "ws" / "_private" / router.PRIVATE_DIR_NAME
    private.mkdir(parents=True)
    assert router.private_overlay_path() == private / "providers.local.json"


def test_unrelated_private_dir_named_like_the_checkout_is_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ``_private/<checkout folder>`` that is not one of the fixed names is
    someone else's inputs, never this package's overlay."""
    pkg = tmp_path / "ws" / "some-other-name" / "router"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(router, "DATA_DIR", pkg)
    monkeypatch.delenv("FIR_INPUTS_DIR", raising=False)
    (tmp_path / "ws" / "_private" / "some-other-name").mkdir(parents=True)
    assert router.private_overlay_path() is None


@pytest.mark.parametrize(
    "value",
    [
        True,
        1,
        {"a": 1},
        "",
        "   ",
        "2026-01-01 JP:",
        "2026-01-01 JP:  ",
        " 2026-01-01 JP: x",
        "2026-01-01 JP: x ",
        "JP: x",
        None,
    ],
)
def test_overlay_rejects_an_attestation_off_the_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: object
) -> None:
    """The schema pattern (``YYYY-MM-DD JP: <text>``) is the contract; the
    overlay fails closed on anything else instead of opening the gate."""
    _overlay(
        tmp_path,
        monkeypatch,
        [
            {
                "provider": GOOGLE[0],
                "model_id": GOOGLE[1],
                "billing_disabled_attested": value,
            },
        ],
    )
    with pytest.raises(ValueError, match="attestation"):
        router.load_providers()


def test_overlay_rejects_duplicate_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    row = {
        "provider": GOOGLE[0],
        "model_id": GOOGLE[1],
        "billing_disabled_attested": "2026-01-01 JP: fixture",
    }
    _overlay(tmp_path, monkeypatch, [row, dict(row)])
    with pytest.raises(ValueError, match="twice"):
        router.load_providers()


def test_overlay_rejects_a_non_dict_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _overlay(tmp_path, monkeypatch, ["google"])
    with pytest.raises(ValueError, match="object"):
        router.load_providers()


@pytest.mark.parametrize("value", [True, 1, "yes", "   ", "2026-01-01 JP: "])
def test_billing_attested_accepts_only_the_contract(value: object) -> None:
    gated = {"billing_gated": True, "billing_disabled_attested": value}
    assert not router.billing_attested(gated)
    assert router.billing_attested(
        {"billing_gated": True, "billing_disabled_attested": "2026-01-01 JP: x"}
    )
