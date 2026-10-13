"""Tests for router.quota — pre-dispatch quota ledger.

Binding rules: docs/POLICY.md:19-28 (zero-spend / caps checked before the
call, never discovered via a 429) and docs/POLICY.md:24 (reset boundary is
null in the registry today, treated as UTC midnight / UTC minute).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from router.quota import QuotaLedger, card_gate, price_is_zero

UTC = timezone.utc


def _entry(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "provider": "openrouter",
        "model_id": "<placeholder>:free",
        "rpm": None,
        "rpd": 50,
        "tpm": None,
        "tpd": None,
        "reset_boundary": None,
    }
    base.update(overrides)
    return base


def test_daily_counter_refuses_at_cap_allows_at_cap_minus_one(tmp_path: Path) -> None:
    """POLICY.md:19-28 — 50 for OpenRouter, 1,000 for Groq (contract test 11)."""
    ledger = QuotaLedger(tmp_path / "ledger.json")
    entry = _entry(rpd=50)
    now = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)

    for _ in range(49):
        ledger.record("openrouter", tokens=0, now=now)
    # 49 committed requests: the 50th (cap-1 -> cap) is allowed.
    assert ledger.check("openrouter", entry, estimated_tokens=0, now=now).allowed

    ledger.record("openrouter", tokens=0, now=now)
    # 50 committed requests: exactly at cap refuses the next one.
    verdict = ledger.check("openrouter", entry, estimated_tokens=0, now=now)
    assert not verdict.allowed
    assert verdict.reason == "rpd"

    groq_entry = _entry(provider="groq", rpd=1000)
    groq_ledger = QuotaLedger(tmp_path / "groq-ledger.json")
    for _ in range(999):
        groq_ledger.record("groq", tokens=0, now=now)
    assert groq_ledger.check("groq", groq_entry, estimated_tokens=0, now=now).allowed
    groq_ledger.record("groq", tokens=0, now=now)
    verdict = groq_ledger.check("groq", groq_entry, estimated_tokens=0, now=now)
    assert not verdict.allowed


def test_tpm_estimate_over_cap_refuses_equal_allows_null_never_refuses(
    tmp_path: Path,
) -> None:
    """Groq's binding limit is TPM (8,000) per POLICY.md:27-28."""
    ledger = QuotaLedger(tmp_path / "ledger.json")
    now = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)
    entry = _entry(provider="groq", tpm=8000)

    assert not ledger.check("groq", entry, estimated_tokens=8001, now=now).allowed
    assert ledger.check("groq", entry, estimated_tokens=8001, now=now).reason == "tpm"
    assert ledger.check("groq", entry, estimated_tokens=8000, now=now).allowed

    null_entry = _entry(provider="groq", tpm=None)
    assert ledger.check(
        "groq", null_entry, estimated_tokens=10_000_000, now=now
    ).allowed


def test_minute_bucket_resets_at_next_utc_minute(tmp_path: Path) -> None:
    ledger = QuotaLedger(tmp_path / "ledger.json")
    entry = _entry(rpm=1, rpd=1000)  # daily cap present but never binding here
    t1 = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 8, 30, 12, 0, 59, tzinfo=UTC)
    t3 = datetime(2026, 8, 30, 12, 1, 0, tzinfo=UTC)

    ledger.record("openrouter", tokens=0, now=t1)
    assert not ledger.check("openrouter", entry, estimated_tokens=0, now=t2).allowed
    assert ledger.check("openrouter", entry, estimated_tokens=0, now=t3).allowed


def test_day_bucket_resets_at_utc_midnight_not_local_midnight(
    tmp_path: Path,
) -> None:
    ledger = QuotaLedger(tmp_path / "ledger.json")
    entry = _entry(rpd=1, rpm=None)
    # 23:30 in UTC-5 is 04:30 UTC the next day — local midnight has not
    # occurred yet in this tz, but UTC midnight has.
    tz_minus5 = timezone(timedelta(hours=-5))
    before_utc_midnight = datetime(2026, 8, 30, 23, 0, 0, tzinfo=UTC)
    after_utc_midnight_local_tz = datetime(
        2026, 8, 30, 20, 1, 0, tzinfo=tz_minus5
    )  # == 2026-08-31 01:01 UTC

    ledger.record("openrouter", tokens=0, now=before_utc_midnight)
    assert not ledger.check(
        "openrouter", entry, estimated_tokens=0, now=before_utc_midnight
    ).allowed
    # New UTC day -> bucket reset even though local-tz day hasn't advanced
    # by a full 24h from the first call's local-equivalent.
    assert ledger.check(
        "openrouter", entry, estimated_tokens=0, now=after_utc_midnight_local_tz
    ).allowed


def test_ledger_survives_simulated_crash(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger = QuotaLedger(path)
    now = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)
    ledger.record("openrouter", tokens=10, now=now)

    # Simulate a crash mid-write: a partial temp file sits next to the
    # ledger, but os.replace never ran, so the committed file is untouched.
    partial = path.with_suffix(path.suffix + ".tmp999999")
    partial.write_text("{not valid json", encoding="utf-8")

    reloaded = QuotaLedger(path)
    entry = _entry(rpd=1, rpm=None)
    verdict = reloaded.check("openrouter", entry, estimated_tokens=0, now=now)
    assert not verdict.allowed  # last committed state (1 request) wins
    assert verdict.reason == "rpd"


def test_two_ledgers_on_same_path_see_each_others_records(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger_a = QuotaLedger(path)
    ledger_b = QuotaLedger(path)
    now = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)
    entry = _entry(rpd=1, rpm=None)

    ledger_a.record("openrouter", tokens=0, now=now)
    # A second process/instance re-loads from disk and must see A's write,
    # so it cannot double-spend the last request.
    verdict = ledger_b.check("openrouter", entry, estimated_tokens=0, now=now)
    assert not verdict.allowed


@pytest.mark.parametrize(
    ("price_in", "price_out", "expected"),
    [
        (0, 0, True),
        (0.0, 0.0, True),
        (None, 0, False),
        (0, None, False),
        (0.0001, 0, False),
        ("0", 0, False),
        (0, "0", False),
    ],
)
def test_price_is_zero(price_in: Any, price_out: Any, expected: bool) -> None:
    entry = {
        "price_usd_per_mtok_in": price_in,
        "price_usd_per_mtok_out": price_out,
    }
    assert price_is_zero(entry) is expected


def test_price_is_zero_missing_keys() -> None:
    assert price_is_zero({}) is False


@pytest.mark.parametrize(
    ("card_required", "expected"),
    [
        ("no", True),
        ("yes", False),
        ("unverified", False),
    ],
)
def test_card_gate(card_required: str, expected: bool) -> None:
    assert card_gate({"card_required": card_required}) is expected


def test_card_gate_missing_key() -> None:
    assert card_gate({}) is False


def test_remote_provider_with_null_daily_cap_is_refused_as_uncapped(
    tmp_path: Path,
) -> None:
    """2026-09-02: google rows shipped with every cap null and the ledger read
    null as unlimited -- a billed key served 180 calls. Null rpd on a remote
    provider now means "no cap recorded", which is a refusal, not a pass."""
    ledger = QuotaLedger(tmp_path / "ledger.json")
    now = datetime(2026, 9, 2, 12, 0, 0, tzinfo=UTC)
    verdict = ledger.check(
        "google", _entry(provider="google", rpd=None), estimated_tokens=1, now=now
    )
    assert not verdict.allowed
    assert verdict.reason == "uncapped"


def test_local_provider_may_run_with_null_caps(tmp_path: Path) -> None:
    ledger = QuotaLedger(tmp_path / "ledger.json")
    now = datetime(2026, 9, 2, 12, 0, 0, tzinfo=UTC)
    entry = _entry(provider="local", rpm=None, rpd=None, tpm=None, tpd=None)
    assert ledger.check("local", entry, estimated_tokens=10_000_000, now=now).allowed
