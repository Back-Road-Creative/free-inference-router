"""Checks on the router's static registry data.

Asserts the data files are internally consistent with their own schema and
with the registry policy decisions (D1/D4: deny by default; training-class
providers restricted to read-only/benchmark roles). Pure stdlib + pytest, no
network, no jsonschema dep (hand-rolled check instead).
"""

import json
from pathlib import Path

ROUTER_DIR = Path(__file__).resolve().parents[1] / "router"
SCHEMA = json.loads((ROUTER_DIR / "registry.schema.json").read_text())
PROVIDERS = json.loads((ROUTER_DIR / "providers.json").read_text())
CLASSIFICATION = json.loads((ROUTER_DIR / "classification.json").read_text())

REQUIRED_KEYS = set(SCHEMA["required"])
ENUMS = {
    k: set(SCHEMA["properties"][k]["enum"])
    for k in ("status", "data_policy_class", "card_required")
}
# Roles permitted for a "training"-class provider (D4): read-only/benchmark only.
TRAINING_SAFE_ROLES = {"synthetic-benchmark", "public-repo-read"}


def test_every_provider_has_required_keys() -> None:
    for entry in PROVIDERS:
        missing = REQUIRED_KEYS - set(entry)
        assert not missing, (
            f"{entry.get('provider')}/{entry.get('model_id')} missing {missing}"
        )


def test_every_provider_enum_field_is_valid() -> None:
    for entry in PROVIDERS:
        for field, allowed in ENUMS.items():
            assert entry[field] in allowed, (
                f"{entry['provider']}/{entry['model_id']} bad {field}={entry[field]!r}"
            )


def test_admitted_set_matches_the_admission_bar_evaluation() -> None:
    """Each row here cleared its smoke AND the Phase-3 admission bars re-derived
    from the 2026-09-05 bench run (results-v2). poolside/laguna-s-2.1
    left the set on 2026-09-05: 43 of its 50 bench calls returned a provider
    error, so no bar had usable evidence -> quarantined, not admitted."""
    admitted = {
        (e["provider"], e["model_id"]) for e in PROVIDERS if e["status"] == "admitted"
    }
    assert admitted == {
        ("groq", "openai/gpt-oss-120b"),
        ("groq", "openai/gpt-oss-20b"),
        ("openrouter", "inclusionai/ling-3.0-flash-fin:free"),
        ("google", "gemini-3.5-flash"),
        ("google", "gemini-3.5-flash-lite"),
        ("local", "qwen3.5:4b"),
    }


def test_no_provider_carries_a_key_the_schema_does_not_declare() -> None:
    """The schema is additionalProperties:false; nothing validated that, so a
    typo'd or undeclared flag used to ride along unnoticed."""
    assert SCHEMA["additionalProperties"] is False
    declared = set(SCHEMA["properties"])
    for e in PROVIDERS:
        extra = set(e) - declared
        assert not extra, f"{e['provider']}/{e['model_id']} carries undeclared {extra}"


def test_rows_that_leaked_on_the_resistance_bar_are_marked_output_is_data() -> None:
    """Resistance bar is zero leaks. These three leaked at least once, so they
    keep their bounded roles but their replies are DATA: only a mechanical
    judge may consume them, never another agent's instruction stream."""
    marked = {
        (e["provider"], e["model_id"]) for e in PROVIDERS if e.get("output_is_data")
    }
    assert marked == {
        ("groq", "openai/gpt-oss-120b"),
        ("groq", "openai/gpt-oss-20b"),
        ("local", "qwen3.5:4b"),
    }


def test_rows_below_the_triage_bar_keep_synthetic_benchmark_only() -> None:
    """Triage bar is >=0.80 agreement. ling scored 0.11 and gemini-3.5-flash
    0.44 (error rows excluded), so neither may read a real public repo."""
    below_bar = (
        ("openrouter", "inclusionai/ling-3.0-flash-fin:free"),
        ("google", "gemini-3.5-flash"),
    )
    for key in below_bar:
        entry = next(e for e in PROVIDERS if (e["provider"], e["model_id"]) == key)
        assert entry["allowed_roles"] == ["synthetic-benchmark"], (
            f"{key} kept {entry['allowed_roles']}"
        )


def test_every_admitted_entry_carries_its_admission_evidence() -> None:
    for e in PROVIDERS:
        if e["status"] != "admitted":
            continue
        who = f"{e['provider']}/{e['model_id']}"
        assert e["last_smoke"], f"{who} admitted without a last_smoke timestamp"
        assert e["card_required"] == "no", (
            f"{who} admitted with card_required={e['card_required']!r}"
        )
        # An admitted entry is either no-training, or training-class locked to
        # the D4 read-only/benchmark roles (the per-class role test enforces
        # the subset; here we refuse "unknown" outright).
        assert e["data_policy_class"] in ("no-training", "training"), (
            f"{who} admitted with unresolved class {e['data_policy_class']!r}"
        )
        assert e["allowed_roles"], f"{who} admitted with no roles"
        assert e["price_usd_per_mtok_in"] == 0 and e["price_usd_per_mtok_out"] == 0, (
            f"{who} admitted with non-zero price"
        )


def test_training_class_providers_are_role_restricted() -> None:
    for entry in PROVIDERS:
        if entry["data_policy_class"] == "training":
            who = f"{entry['provider']}/{entry['model_id']}"
            roles = set(entry["allowed_roles"])
            assert roles <= TRAINING_SAFE_ROLES, (
                f"{who} allows {roles - TRAINING_SAFE_ROLES}"
            )
            # Non-emptiness is an *admission* property: a quarantined row holds
            # no roles at all (poolside, 2026-09-05).
            if entry["status"] == "admitted":
                assert roles, f"{who} admitted with no roles"


def test_classification_defaults_to_deny() -> None:
    assert CLASSIFICATION["default"] == "deny"


def test_every_non_null_price_is_zero() -> None:
    for entry in PROVIDERS:
        for field in ("price_usd_per_mtok_in", "price_usd_per_mtok_out"):
            price = entry[field]
            assert price is None or price == 0, (
                f"{entry['provider']} {field}={price} is not zero"
            )


#: Providers whose "free tier" exists only while the key's cloud project has NO
#: billing account linked. Code cannot see that; a human attests it, by project.
BILLING_GATED_PROVIDERS = {"google"}


def test_admitted_remote_rows_carry_a_daily_cap() -> None:
    """2026-09-02: both google rows were admitted with rpd null; the ledger read
    null as unlimited and a billed key served 180 calls. Remote = capped."""
    for e in PROVIDERS:
        if e["status"] != "admitted" or e["provider"] == "local":
            continue
        assert e["rpd"] is not None, (
            f"{e['provider']}/{e['model_id']} admitted with no rpd"
        )


def test_billing_gated_rows_are_flagged_and_unattested_in_the_tracked_file() -> None:
    """2026-10-14: the attestation names the operator's cloud project, so it
    lives out of the tracked registry, in the operator overlay
    (providers.local.json under FIR_INPUTS_DIR or
    _private/free-inference-router). The tracked row says
    ``billing_gated: true`` and carries a null attestation;
    router.selector.provider_eligible and the bench runner refuse such a row
    until the private overlay attests it (tests/test_selector.py,
    tests/test_bench_runner.py)."""
    for e in PROVIDERS:
        assert e.get("billing_disabled_attested") is None, (
            f"{e['provider']}/{e['model_id']} carries an attestation "
            "in the tracked file"
        )
        if e["provider"] in BILLING_GATED_PROVIDERS:
            assert e.get("billing_gated") is True, (
                f"{e['provider']}/{e['model_id']} not billing_gated"
            )
