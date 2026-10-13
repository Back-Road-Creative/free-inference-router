"""Tests for router.candidates -- laya-candidate/v1 schema, grouped
splits, preregistered baselines (Y0, plan §3.7). Offline/stdlib only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import router.decide as decide_module
from router.candidates import (
    CANDIDATE_SCHEMA,
    CLASS_FLOOR,
    CandidateRow,
    MajorityBaseline,
    PreregError,
    RuleBaseline,
    class_count_report,
    enforce_token_budget,
    ensure_test_split_authorized,
    feature_builder_sha256,
    fit_majority,
    load_candidates,
    load_prereg,
    score_predictions,
    split,
    state_text,
    validate_candidate_dict,
)


def _row(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema": CANDIDATE_SCHEMA,
        "task_type": "ci-triage",
        "candidate_id": "cand-1",
        "ts": "2026-09-20T00:00:00Z",
        "group_id": "pr-1",
        "features": {"n_files": 3, "workflow": "CI"},
        "options": ["infra", "regression"],
        "label": "infra",
        "label_provenance": "adjudicated",
        "censored": False,
        "outcome": "failed",
    }
    base.update(overrides)
    return base


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def test_validate_accepts_a_good_row() -> None:
    assert validate_candidate_dict(_row()) is None
    assert validate_candidate_dict(_row(label=None)) is None


@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        ({"schema": "laya-candidate/v2"}, "wrong_schema"),
        ({"task_type": ""}, "bad_task_type"),
        ({"candidate_id": 5}, "bad_candidate_id"),
        ({"group_id": ""}, "bad_group_id"),
        ({"ts": ""}, "bad_ts"),
        ({"features": "not-a-dict"}, "bad_features"),
        ({"features": {"k": ["nested", "list"]}}, "bad_features"),
        ({"options": []}, "bad_options"),
        ({"options": ["a", "a"]}, "bad_options"),
        ({"options": ["a", "b"], "label": "c"}, "bad_label"),
        ({"label_provenance": "guessed"}, "bad_label_provenance"),
        ({"censored": "no"}, "bad_censored"),
        ({"outcome": 12}, "bad_outcome"),
    ],
)
def test_validate_rejects_each_bad_field(
    overrides: dict[str, Any], expected_code: str
) -> None:
    assert validate_candidate_dict(_row(**overrides)) == expected_code


def test_validate_rejects_missing_field() -> None:
    bad = _row()
    del bad["outcome"]
    assert validate_candidate_dict(bad) == "missing_field:outcome"


def test_validate_rejects_non_object() -> None:
    assert validate_candidate_dict(["not", "an", "object"]) == "not_object"


def test_load_candidates_parses_valid_rows(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "c.jsonl", [_row(candidate_id="c1"), _row(candidate_id="c2")]
    )
    rows, bad_counts = load_candidates(path)
    assert len(rows) == 2
    assert sum(bad_counts.values()) == 0
    assert isinstance(rows[0], CandidateRow)


def test_load_candidates_counts_each_bad_row_by_code(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps(_row(candidate_id="good")) + "\n"
        "{not json\n"
        + json.dumps(_row(candidate_id="bad-schema", schema="other/v1"))
        + "\n"
        + json.dumps({**_row(candidate_id="bad-options"), "options": []})
        + "\n"
    )
    rows, bad_counts = load_candidates(path)
    assert len(rows) == 1
    assert bad_counts["bad_json"] == 1
    assert bad_counts["wrong_schema"] == 1
    assert bad_counts["bad_options"] == 1
    assert sum(bad_counts.values()) == 3


def test_load_candidates_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps(_row()) + "\n\n\n")
    rows, bad_counts = load_candidates(path)
    assert len(rows) == 1
    assert sum(bad_counts.values()) == 0


def _rows_across_groups(n_groups: int, per_group: int = 3) -> list[CandidateRow]:
    rows = []
    for g in range(n_groups):
        for i in range(per_group):
            data = _row(
                candidate_id=f"g{g}-{i}",
                group_id=f"grp-{g}",
                ts=f"2026-09-{(g % 28) + 1:02d}T00:00:{i:02d}Z",
            )
            rows.append(CandidateRow.from_dict(data))
    return rows


def test_split_never_puts_one_group_on_two_sides() -> None:
    rows = _rows_across_groups(20)
    out = split(rows, dev_frac=0.6, cal_frac=0.2, seed=7)
    dev_groups = {r.group_id for r in out["dev"]}
    cal_groups = {r.group_id for r in out["cal"]}
    test_groups = {r.group_id for r in out["test"]}
    assert not (dev_groups & cal_groups)
    assert not (dev_groups & test_groups)
    assert not (cal_groups & test_groups)
    assert sum(len(v) for v in out.values()) == len(rows)


def test_split_orders_by_time_dev_holds_earliest() -> None:
    rows = _rows_across_groups(10)
    out = split(rows, dev_frac=0.3, cal_frac=0.3, seed=1)
    assert max(r.ts for r in out["dev"]) <= min(r.ts for r in out["test"])


def test_split_rejects_bad_fractions() -> None:
    with pytest.raises(ValueError):
        split(_rows_across_groups(3), dev_frac=0.8, cal_frac=0.5, seed=0)


def test_class_count_report_flags_evaluable_at_floor() -> None:
    rows = [
        CandidateRow.from_dict(_row(candidate_id=f"a{i}", label="infra"))
        for i in range(CLASS_FLOOR)
    ] + [
        CandidateRow.from_dict(_row(candidate_id=f"b{i}", label="regression"))
        for i in range(CLASS_FLOOR)
    ]
    report = class_count_report(rows)
    assert report["evaluable"] is True
    assert report["class_counts"] == {"infra": CLASS_FLOOR, "regression": CLASS_FLOOR}


def test_class_count_report_flags_insufficient_below_floor() -> None:
    rows = [
        CandidateRow.from_dict(_row(candidate_id=f"a{i}", label="infra"))
        for i in range(CLASS_FLOOR)
    ] + [CandidateRow.from_dict(_row(candidate_id="b0", label="regression"))]
    report = class_count_report(rows)
    assert report["evaluable"] is False
    assert report["class_counts"]["regression"] == 1


def _write_prereg(path: Path, builder_hash: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "feature_builder_sha256": builder_hash,
                "candidates": {
                    "y1a": {
                        "metrics": ["precision"],
                        "false_negative_limit": 0.05,
                        "coverage_floor": 0.5,
                        "latency_budget_s": 1.0,
                    }
                },
            }
        )
    )
    return path


@pytest.mark.parametrize(
    "prereg_data",
    [
        {"candidates": {}},
        {"feature_builder_sha256": "x", "candidates": {"y1a": {"metrics": []}}},
    ],
)
def test_load_prereg_rejects_malformed(
    tmp_path: Path, prereg_data: dict[str, Any]
) -> None:
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(prereg_data))
    with pytest.raises(PreregError):
        load_prereg(path)


_BUILDER_PATH = Path(__file__).parents[1] / "router" / "candidates.py"


def test_ensure_test_split_authorized_refuses_without_prereg_file() -> None:
    with pytest.raises(PreregError):
        ensure_test_split_authorized(None, _BUILDER_PATH)


def test_ensure_test_split_authorized_refuses_on_hash_mismatch(tmp_path: Path) -> None:
    prereg_path = _write_prereg(tmp_path / "prereg.json", builder_hash="deadbeef")
    with pytest.raises(PreregError, match="hash mismatch"):
        ensure_test_split_authorized(prereg_path, _BUILDER_PATH)


def test_ensure_test_split_authorized_passes_on_matching_hash(tmp_path: Path) -> None:
    real_hash = feature_builder_sha256(_BUILDER_PATH)
    prereg_path = _write_prereg(tmp_path / "prereg.json", builder_hash=real_hash)
    data = ensure_test_split_authorized(prereg_path, _BUILDER_PATH)
    assert data["feature_builder_sha256"] == real_hash


def test_state_text_is_compact_and_sorted() -> None:
    row = CandidateRow.from_dict(_row(features={"b": 2, "a": 1}))
    assert state_text(row) == "a=1; b=2"


def _approx4(t: str) -> int:
    return len(t) // 4


def test_enforce_token_budget_flags_truncation() -> None:
    text = "x" * 1000
    truncated_text, truncated = enforce_token_budget(
        text, max_tokens=10, approx_tokens=_approx4
    )
    assert truncated is True
    assert _approx4(truncated_text) <= 10
    assert len(truncated_text) < len(text)

    short = "short state"
    short_text, short_truncated = enforce_token_budget(
        short, max_tokens=100, approx_tokens=_approx4
    )
    assert short_truncated is False
    assert short_text == short


def test_fit_majority_picks_most_common_label() -> None:
    rows = [
        CandidateRow.from_dict(_row(candidate_id="a", label="infra")),
        CandidateRow.from_dict(_row(candidate_id="b", label="infra")),
        CandidateRow.from_dict(_row(candidate_id="c", label="regression")),
    ]
    label = fit_majority(rows)
    assert label == "infra"
    baseline = MajorityBaseline(label=label)
    assert baseline.predict(CandidateRow.from_dict(_row(label=None))) == "infra"


def test_rule_baseline_always_first_option() -> None:
    baseline = RuleBaseline("always_first_option")
    row = CandidateRow.from_dict(_row(options=["z", "a"], label=None))
    assert baseline.predict(row) == "z"


def test_score_predictions_confusion_and_balanced_accuracy() -> None:
    rows = [
        CandidateRow.from_dict(_row(candidate_id="a", label="infra")),
        CandidateRow.from_dict(_row(candidate_id="b", label="regression")),
    ]

    def predict(row: CandidateRow) -> str:
        return "infra"

    report = score_predictions(predict, rows, ["infra", "regression"])
    assert report["confusion"]["infra"]["infra"] == 1
    assert report["confusion"]["regression"]["infra"] == 1
    assert report["per_class"]["regression"]["recall"] == 0.0
    assert report["balanced_accuracy"] == pytest.approx(0.5)


def test_cli_candidates_and_labels_are_mutually_exclusive(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text("")
    candidates_path = _write_jsonl(tmp_path / "cands.jsonl", [_row()])
    with pytest.raises(SystemExit):
        decide_module.main(
            [
                "--labels",
                str(labels_path),
                "--candidates",
                str(candidates_path),
                "--scorer",
                "fixed:infra",
            ]
        )


def test_cli_requires_labels_or_candidates() -> None:
    with pytest.raises(SystemExit):
        decide_module.main([])


def test_cli_candidates_test_split_refuses_without_prereg(tmp_path: Path) -> None:
    candidates_path = _write_jsonl(tmp_path / "cands.jsonl", [_row()])
    with pytest.raises(PreregError):
        decide_module.main(
            ["--candidates", str(candidates_path), "--split", "test", "--json"]
        )


def test_cli_candidates_json_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    small_path = _write_jsonl(tmp_path / "small.jsonl", [_row()])
    assert decide_module.main(["--candidates", str(small_path), "--json"]) == 0
    small = json.loads(capsys.readouterr().out)
    assert small["n_rows"] == 1
    assert small["insufficient_data"] is True

    rows = []
    for i in range(25):
        rows.append(_row(candidate_id=f"a{i}", group_id=f"a{i}", label="infra"))
        rows.append(_row(candidate_id=f"b{i}", group_id=f"b{i}", label="regression"))
    big_path = _write_jsonl(tmp_path / "big.jsonl", rows)
    assert decide_module.main(["--candidates", str(big_path), "--json"]) == 0
    big = json.loads(capsys.readouterr().out)
    assert big["schema"] == CANDIDATE_SCHEMA
    assert big["n_rows"] == 50
    assert big["laya"] is None
    assert "majority" in big["baselines"]
