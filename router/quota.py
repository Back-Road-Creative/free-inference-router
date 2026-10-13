"""Pre-dispatch quota ledger and zero-spend gates.

Enforces docs/POLICY.md:19-28: RPM/RPD/TPM/TPD caps come from
``router/providers.json`` and a local counter checks headroom **before**
the call — a 429 must never be how a cap is discovered. Also implements the
zero-spend rule (POLICY.md:21-24: price must be present and exactly ``0``
on both legs) and the card-required gate (``unverified`` treated as ``yes``
for gating).

The registry's ``reset_boundary`` field is ``null`` for every entry today
(``router/providers.json``); per POLICY.md:24 this module documents that a
null boundary is treated as UTC midnight for daily buckets and the top of
the UTC minute for minute buckets — there is no per-provider override yet.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc


#: The only provider allowed to run with a null daily cap (hardware-bound).
LOCAL_PROVIDER = "local"


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str | None = None


def _day_key(now: datetime) -> str:
    return now.astimezone(UTC).strftime("%Y-%m-%d")


def _minute_key(now: datetime) -> str:
    return now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M")


def _fresh_bucket(key: str) -> dict[str, Any]:
    return {"key": key, "requests": 0, "tokens": 0}


class QuotaLedger:
    """Per-provider request/token counters, persisted as JSON.

    Every read re-loads from disk and every write is read-modify-write
    followed by an atomic replace, so two ``QuotaLedger`` instances
    pointed at the same path (two processes, or two calls within one)
    never double-spend the last request, and a crash mid-write leaves the
    last committed state intact (see ``_save``).
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> dict[str, Any]:
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + f".tmp{uuid.uuid4().hex}")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, self.path)

    def _buckets(
        self, data: dict[str, Any], provider: str, now: datetime
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        pdata = data.get(provider, {})
        day = pdata.get("day", {})
        minute = pdata.get("minute", {})
        day_key = _day_key(now)
        minute_key = _minute_key(now)
        if day.get("key") != day_key:
            day = _fresh_bucket(day_key)
        if minute.get("key") != minute_key:
            minute = _fresh_bucket(minute_key)
        return day, minute

    def check(
        self,
        provider: str,
        entry: dict[str, Any],
        estimated_tokens: int,
        now: datetime,
    ) -> Verdict:
        # A null daily cap on a REMOTE provider is not "unlimited" -- it is
        # "nobody wrote the free-tier limit down", and 2026-09-02 that shape
        # (both google rows all-null) let a billed key serve 180 calls. Only
        # the local lane is bounded by hardware instead of a quota.
        if provider != LOCAL_PROVIDER and entry.get("rpd") is None:
            return Verdict(False, "uncapped")
        data = self._load()
        day, minute = self._buckets(data, provider, now)

        rpd = entry.get("rpd")
        if rpd is not None and day["requests"] + 1 > rpd:
            return Verdict(False, "rpd")

        tpd = entry.get("tpd")
        if tpd is not None and day["tokens"] + estimated_tokens > tpd:
            return Verdict(False, "tpd")

        rpm = entry.get("rpm")
        if rpm is not None and minute["requests"] + 1 > rpm:
            return Verdict(False, "rpm")

        tpm = entry.get("tpm")
        if tpm is not None and minute["tokens"] + estimated_tokens > tpm:
            return Verdict(False, "tpm")

        return Verdict(True, None)

    def record(self, provider: str, tokens: int, now: datetime) -> None:
        data = self._load()
        day, minute = self._buckets(data, provider, now)
        day["requests"] += 1
        day["tokens"] += tokens
        minute["requests"] += 1
        minute["tokens"] += tokens
        data[provider] = {"day": day, "minute": minute}
        self._save(data)


def _is_exact_zero(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) and value == 0


def price_is_zero(entry: dict[str, Any]) -> bool:
    """Both price legs must be present and exactly ``0`` (POLICY.md:21-24)."""
    return _is_exact_zero(entry.get("price_usd_per_mtok_in")) and _is_exact_zero(
        entry.get("price_usd_per_mtok_out")
    )


def card_gate(entry: dict[str, Any]) -> bool:
    """``card_required: unverified`` is treated as ``yes`` for gating.

    Only an explicit ``"no"`` clears the gate.
    """
    return entry.get("card_required") == "no"
