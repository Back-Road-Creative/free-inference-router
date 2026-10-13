"""Property-based tests for router.quarantine (hypothesis).

Contract under test, cited from the module and the policy it implements:

- router/quarantine.py:1,19 — the ledger is APPEND-ONLY JSONL; ``record``
  (quarantine.py:24-33) only ever appends one line per call.
- router/quarantine.py:35-48 — ``is_quarantined`` is True iff SOME line
  matches BOTH ``provider`` and ``model_id`` (line 46); it never reads
  ``at`` or ``reason``, so a record never expires and there is no clear
  API. docs/POLICY.md:82-83,86: a quarantine holds "until JP re-approves"
  and "stays JP's decision" — the module deliberately has no expiry.
- router/quarantine.py:6-8 and docs/THREAT-MODEL.md:23-30 — a record
  carries only identifiers, a reason string, and a timestamp.

The observable state is therefore the function
``(provider, model_id) -> is_quarantined``, which must equal
``pair in {pairs recorded so far}`` regardless of order, repetition,
reason, or timestamp.  The serialized form is NOT idempotent by design
(append-only), so the properties pin the line count to the call count.

Temp dirs are opened inside each test body: hypothesis forbids
function-scoped fixtures such as ``tmp_path`` under ``@given``.
``database=None`` keeps hypothesis from creating an untracked
``.hypothesis/`` directory next to the tests.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from router.quarantine import Quarantine

UTC = timezone.utc
T0 = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)

Pair = tuple[str, str]
# (True -> record the pair, False -> only query it)
Op = tuple[bool, Pair]

ident = st.text(min_size=1, max_size=12)
pair_st: st.SearchStrategy[Pair] = st.tuples(ident, ident)
pairs_st = st.lists(pair_st, max_size=8)
reason_st = st.text(max_size=20)
at_st = st.datetimes(
    timezones=st.sampled_from([None, UTC, timezone(timedelta(hours=-5))])
)

PROPERTY = settings(deadline=None, max_examples=200, database=None)


@st.composite
def op_sequences(draw: st.DrawFn) -> tuple[list[Pair], list[Op]]:
    """A small pool of pairs and a sequence of record/query ops over it.

    Ops cross the pool's providers with its models so repeats, "same
    provider, other model" and "same model, other provider" probes are
    common instead of vanishingly rare.
    """
    pool = draw(st.lists(pair_st, min_size=1, max_size=5, unique=True))
    providers = [p for p, _ in pool]
    models = [m for _, m in pool]
    crossed = st.tuples(st.sampled_from(providers), st.sampled_from(models))
    ops = draw(st.lists(st.tuples(st.booleans(), crossed), max_size=25))
    return pool, ops


def _ledger(path: Path, pairs: list[Pair]) -> Quarantine:
    q = Quarantine(path)
    for provider, model_id in pairs:
        q.record(provider, model_id, "model_mismatch", at=T0)
    return q


def _state(q: Quarantine, probes: list[Pair]) -> list[bool]:
    return [q.is_quarantined(p, m) for p, m in probes]


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


@PROPERTY
@given(pairs=pairs_st, extra=pair_st)
def test_property_quarantine_twice_equals_once(pairs: list[Pair], extra: Pair) -> None:
    """Recording a pair again changes nothing observable through
    ``is_quarantined`` (quarantine.py:46 — any matching line suffices) while
    the append-only ledger grows by exactly one line (quarantine.py:31-33)."""
    with tempfile.TemporaryDirectory() as d:
        once = _ledger(Path(d) / "once.jsonl", [*pairs, extra])
        twice = _ledger(Path(d) / "twice.jsonl", [*pairs, extra, extra])
        provider, model_id = extra
        probes = [*pairs, extra, (provider, model_id + "x"), (provider + "x", model_id)]

        assert _state(once, probes) == _state(twice, probes)
        assert once.is_quarantined(*extra) is True
        assert _line_count(once.path) == len(pairs) + 1
        assert _line_count(twice.path) == len(pairs) + 2


@PROPERTY
@given(pairs=pairs_st, probe=pair_st)
def test_property_unrecorded_pair_is_isolated(pairs: list[Pair], probe: Pair) -> None:
    """A pair never recorded reports False no matter what else was recorded
    — including its provider under another model and its model under
    another provider (quarantine.py:46 matches BOTH fields)."""
    recorded = [p for p in pairs if p != probe]
    with tempfile.TemporaryDirectory() as d:
        q = _ledger(Path(d) / "q.jsonl", recorded)
        assert q.is_quarantined(*probe) is False

        provider, model_id = probe
        other_model = (provider, model_id + "!")
        other_provider = (provider + "!", model_id)
        assert q.is_quarantined(*other_model) is (other_model in recorded)
        assert q.is_quarantined(*other_provider) is (other_provider in recorded)


@PROPERTY
@given(pairs=pairs_st, data=st.data())
def test_property_state_is_order_independent(
    pairs: list[Pair], data: st.DataObject
) -> None:
    """The observable state is a function of the SET of recorded pairs, not
    the order they arrived in (quarantine.py:41-47 scans every line)."""
    shuffled = data.draw(st.permutations(pairs))
    probes = [*pairs, ("never", "recorded")]
    with tempfile.TemporaryDirectory() as d:
        a = _ledger(Path(d) / "a.jsonl", pairs)
        b = _ledger(Path(d) / "b.jsonl", shuffled)
        assert _state(a, probes) == _state(b, probes)
        assert _state(a, probes) == [p in set(pairs) for p in probes]


@PROPERTY
@given(pair=pair_st, reason=reason_st, at=at_st)
def test_property_no_expiry_any_timestamp_stays_quarantined(
    pair: Pair, reason: str, at: datetime
) -> None:
    """There is no expiry: ``is_quarantined`` never reads ``at`` or
    ``reason`` (quarantine.py:35-48), and the module exposes no clear API —
    POLICY.md:82-83,86: quarantined "until JP re-approves"; "quarantine
    itself stays JP's decision". Whatever the timestamp (past, future,
    naive, aware) the pair stays quarantined; a time-based expiry added
    later would have to read ``at`` and would break this property."""
    with tempfile.TemporaryDirectory() as d:
        q = Quarantine(Path(d) / "q.jsonl")
        q.record(pair[0], pair[1], reason, at=at)
        assert q.is_quarantined(*pair) is True
    assert not [name for name in dir(q) if "clear" in name or "expire" in name], (
        "quarantine.py:1 is append-only; a clear/expiry API would break it"
    )


@PROPERTY
@given(pair=pair_st, reason=reason_st, at=at_st)
def test_property_record_carries_only_identifiers(
    pair: Pair, reason: str, at: datetime
) -> None:
    """Each line is exactly ``{provider, model_id, reason, at}`` with the
    inputs round-tripped (quarantine.py:25-30; THREAT-MODEL.md:23-30: never
    prompt or response content), and the timestamp survives ``isoformat``."""
    with tempfile.TemporaryDirectory() as d:
        q = Quarantine(Path(d) / "q.jsonl")
        q.record(pair[0], pair[1], reason, at=at)
        lines = q.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert set(entry) == {"provider", "model_id", "reason", "at"}
    assert (entry["provider"], entry["model_id"]) == pair
    assert entry["reason"] == reason
    assert datetime.fromisoformat(entry["at"]) == at


@PROPERTY
@given(seq=op_sequences())
def test_property_operation_sequence_tracks_recorded_set(
    seq: tuple[list[Pair], list[Op]],
) -> None:
    """Under any interleaving of record and query, ``is_quarantined`` after
    each step equals membership in the set recorded so far, and the ledger
    holds one line per ``record`` call (quarantine.py:19 append-only)."""
    pool, ops = seq
    recorded: set[Pair] = set()
    calls = 0
    with tempfile.TemporaryDirectory() as d:
        q = Quarantine(Path(d) / "q.jsonl")
        for do_record, (provider, model_id) in ops:
            if do_record:
                q.record(provider, model_id, "terms_change", at=T0)
                recorded.add((provider, model_id))
                calls += 1
            assert q.is_quarantined(provider, model_id) is (
                (provider, model_id) in recorded
            )
            assert _line_count(q.path) == calls
        for p, m in pool:
            assert q.is_quarantined(p, m) is ((p, m) in recorded)
