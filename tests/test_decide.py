"""Tests for router.decide -- offline evaluation harness for a candidate
scorer (Laya or otherwise) against a labelled dispatch-outcome JSONL.

This module is OFFLINE-EVAL ONLY (JP decision D1, 2026-09-22): it never
feeds a decision back into router.selector, permission checks, or a
release. tests/test_decide_not_wired.py asserts router.selector does not
import it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from router.decide import (
    Decision,
    EvalReport,
    FixedScorer,
    LabelledRow,
    evaluate,
    load_labels,
)


def _row(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema": "dispatch-outcome/v1",
        "session": "sess-1",
        "tool_use_id": "tu-1",
        "ts": "2026-09-22T00:00:00Z",
        "subagent_type": "general-purpose",
        "model_param": "sonnet",
        "background": False,
        "prompt_sha": "abc123",
        "prompt_chars": 42,
        "prompt": "do the thing",
        "outcome": "ok",
        "redispatched_by": None,
        "tier_label": None,
    }
    base.update(overrides)
    return base


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


class FakeScorer:
    """Deterministic fake used by every test -- never touches onnxruntime."""

    def __init__(self, tier: str | None = "sonnet", confidence: float = 0.9) -> None:
        self.tier = tier
        self.confidence = confidence
        self.calls: list[str] = []

    def predict(self, prompt: str) -> Decision:
        self.calls.append(prompt)
        return Decision(tier=self.tier, confidence=self.confidence)


# --- load_labels -------------------------------------------------------


def test_load_labels_parses_valid_rows(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "labels.jsonl", [_row(), _row(tool_use_id="tu-2")])
    rows, bad_count = load_labels(path)
    assert len(rows) == 2
    assert bad_count == 0
    assert isinstance(rows[0], LabelledRow)
    assert rows[0].prompt == "do the thing"


def test_load_labels_skips_and_counts_wrong_schema(tmp_path: Path) -> None:
    good = _row()
    bad = _row(schema="dispatch-outcome/v2")
    path = _write_jsonl(tmp_path / "labels.jsonl", [good, bad])
    rows, bad_count = load_labels(path)
    assert len(rows) == 1
    assert bad_count == 1


def test_load_labels_skips_and_counts_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(_row()) + "\n" + "{not json\n")
    rows, bad_count = load_labels(path)
    assert len(rows) == 1
    assert bad_count == 1


def test_load_labels_skips_and_counts_missing_required_field(tmp_path: Path) -> None:
    good = _row()
    bad = _row()
    del bad["outcome"]
    path = _write_jsonl(tmp_path / "labels.jsonl", [good, bad])
    rows, bad_count = load_labels(path)
    assert len(rows) == 1
    assert bad_count == 1


def test_load_labels_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(_row()) + "\n\n\n")
    rows, bad_count = load_labels(path)
    assert len(rows) == 1
    assert bad_count == 0


# --- evaluate ------------------------------------------------------------


def test_evaluate_counts_rows_used_and_excludes_missing_prompt(tmp_path: Path) -> None:
    rows = [
        LabelledRow.from_dict(_row()),
        LabelledRow.from_dict(_row(prompt="", tool_use_id="tu-2")),
        LabelledRow.from_dict(_row(prompt=None, tool_use_id="tu-3")),
    ]
    scorer = FakeScorer(tier="sonnet")
    report = evaluate(rows, scorer)
    assert report.n_used == 1
    assert report.n_excluded_no_prompt == 2


def test_evaluate_reports_abstention_rate(tmp_path: Path) -> None:
    rows = [LabelledRow.from_dict(_row(tool_use_id=f"tu-{i}")) for i in range(4)]
    scorer = FakeScorer(tier=None)  # always abstains
    report = evaluate(rows, scorer)
    assert report.abstention_rate == 1.0


def test_evaluate_agreement_with_baseline_model_param(tmp_path: Path) -> None:
    rows = [
        LabelledRow.from_dict(_row(model_param="sonnet")),
        LabelledRow.from_dict(_row(model_param="opus", tool_use_id="tu-2")),
    ]
    scorer = FakeScorer(tier="sonnet")
    report = evaluate(rows, scorer)
    # baseline default is model_param: agrees with row 1, disagrees with row 2
    assert report.agreement_with_baseline == pytest.approx(0.5)


def test_evaluate_agreement_with_explicit_baseline(tmp_path: Path) -> None:
    rows = [LabelledRow.from_dict(_row(model_param="opus"))]
    scorer = FakeScorer(tier="opus")
    report = evaluate(rows, scorer, baseline="model_param")
    assert report.agreement_with_baseline == 1.0


def test_evaluate_accuracy_vs_tier_label(tmp_path: Path) -> None:
    rows = [
        LabelledRow.from_dict(_row(tier_label="sonnet")),
        LabelledRow.from_dict(_row(tier_label="opus", tool_use_id="tu-2")),
        LabelledRow.from_dict(_row(tier_label=None, tool_use_id="tu-3")),
    ]
    scorer = FakeScorer(tier="sonnet")
    report = evaluate(rows, scorer)
    assert report.n_labelled == 2
    assert report.accuracy_vs_label == pytest.approx(0.5)


def test_evaluate_accuracy_vs_label_zero_labelled_is_fine(tmp_path: Path) -> None:
    rows = [LabelledRow.from_dict(_row(tier_label=None))]
    scorer = FakeScorer(tier="sonnet")
    report = evaluate(rows, scorer)
    assert report.n_labelled == 0
    assert report.accuracy_vs_label is None


def test_evaluate_breakdown_by_outcome(tmp_path: Path) -> None:
    rows = [
        LabelledRow.from_dict(_row(outcome="ok")),
        LabelledRow.from_dict(_row(outcome="ok", tool_use_id="tu-2")),
        LabelledRow.from_dict(_row(outcome="error", tool_use_id="tu-3")),
    ]
    scorer = FakeScorer(tier="sonnet")
    report = evaluate(rows, scorer)
    assert report.breakdown_by_outcome == {"ok": 2, "error": 1}


def test_evaluate_reports_latency_percentiles(tmp_path: Path) -> None:
    rows = [LabelledRow.from_dict(_row(tool_use_id=f"tu-{i}")) for i in range(5)]
    scorer = FakeScorer(tier="sonnet")
    report = evaluate(rows, scorer)
    assert report.predict_latency_p50_s >= 0.0
    assert report.predict_latency_p95_s >= report.predict_latency_p50_s


def test_evaluate_report_is_json_serialisable(tmp_path: Path) -> None:
    rows = [LabelledRow.from_dict(_row())]
    scorer = FakeScorer(tier="sonnet")
    report = evaluate(rows, scorer)
    assert isinstance(report, EvalReport)
    json.dumps(report.to_dict())  # must not raise


# --- FixedScorer -----------------------------------------------------------


def test_fixed_scorer_returns_configured_tier() -> None:
    scorer = FixedScorer(tier="haiku")
    decision = scorer.predict("anything")
    assert decision.tier == "haiku"
    assert decision.confidence == 1.0


def test_fixed_scorer_none_tier_abstains() -> None:
    scorer = FixedScorer(tier=None)
    decision = scorer.predict("anything")
    assert decision.tier is None


# --- OnnxLayaScorer: lazy import, no onnxruntime needed for the suite -----


def test_onnx_laya_scorer_importable_without_onnxruntime(tmp_path: Path) -> None:
    """Importing router.decide must never require onnxruntime -- only
    instantiating OnnxLayaScorer does, and only when actually invoked."""
    import router.decide as decide_module

    assert hasattr(decide_module, "OnnxLayaScorer")


def test_onnx_laya_scorer_raises_clear_error_without_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    from router.decide import OnnxLayaScorer

    fake_model = tmp_path / "laya.onnx"
    fake_model.write_bytes(b"not a real model")

    real_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "onnxruntime":
            raise ImportError("onnxruntime not installed (test double)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError, match="onnxruntime"):
        OnnxLayaScorer(model_path=fake_model)


# --- CLI -------------------------------------------------------------------


def test_cli_with_fixed_scorer_runs_end_to_end(tmp_path: Path) -> None:
    labels_path = _write_jsonl(
        tmp_path / "labels.jsonl",
        [_row(tier_label="sonnet"), _row(tool_use_id="tu-2", tier_label="opus")],
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "router.decide",
            "--labels",
            str(labels_path),
            "--scorer",
            "fixed:sonnet",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["n_used"] == 2
    assert payload["n_labelled"] == 2


def test_cli_requires_model_or_fixed_scorer(tmp_path: Path) -> None:
    labels_path = _write_jsonl(tmp_path / "labels.jsonl", [_row()])
    result = subprocess.run(
        [sys.executable, "-m", "router.decide", "--labels", str(labels_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_main_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import router.decide as decide_module

    labels_path = _write_jsonl(tmp_path / "labels.jsonl", [_row()])
    rc = decide_module.main(
        ["--labels", str(labels_path), "--scorer", "fixed:opus", "--json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_used"] == 1


def test_main_text_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import router.decide as decide_module

    labels_path = _write_jsonl(tmp_path / "labels.jsonl", [_row()])
    rc = decide_module.main(["--labels", str(labels_path), "--scorer", "fixed:none"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "n_used: 1" in out


def test_main_raises_without_model_or_scorer(tmp_path: Path) -> None:
    import router.decide as decide_module

    labels_path = _write_jsonl(tmp_path / "labels.jsonl", [_row()])
    with pytest.raises(SystemExit):
        decide_module.main(["--labels", str(labels_path)])


# --- Laya sequence layout (pure, no runtime) ------------------------------

_SPECIAL = {"[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[PAD]": 0}


def _byte_encode(text: str) -> list[int]:
    """Stand-in tokenizer: one id per byte, offset past the special ids."""
    return [10 + b for b in text.encode("utf-8")]


def _special_ids() -> Any:
    from router.decide import LayaSpecialIds

    return LayaSpecialIds(cls=1, sep=2, mask=3, pad=0)


def test_build_laya_sequence_layout_and_markers() -> None:
    from router.decide import build_laya_sequence

    ids, markers = build_laya_sequence(
        _byte_encode,
        _special_ids(),
        state="the state",
        instructions="which?",
        options=["a: first", "b: second"],
        max_len=512,
        head_max_len=192,
    )
    head = _byte_encode("choice question: which?")
    assert ids[0] == 1 and ids[1 : 1 + len(head)] == head
    assert ids[1 + len(head)] == 2
    assert len(markers) == 2
    assert all(ids[m] == 3 for m in markers)
    first = _byte_encode(" a: first")
    assert ids[markers[0] + 1 :][: len(first)] == first
    assert ids[-1] == 2
    state_start = ids.index(2, markers[1]) + 1
    assert ids[state_start:-1] == _byte_encode("the state")


def test_build_laya_sequence_truncates_state_to_max_len() -> None:
    from router.decide import build_laya_sequence

    ids, markers = build_laya_sequence(
        _byte_encode,
        _special_ids(),
        state="x" * 5000,
        instructions="q",
        options=["a", "b"],
        max_len=64,
        head_max_len=32,
    )
    assert len(ids) == 64
    assert ids[-1] == 2
    assert markers and all(m < 64 for m in markers)


def test_build_laya_sequence_scrubs_mask_token_from_user_text() -> None:
    from router.decide import build_laya_sequence

    ids, markers = build_laya_sequence(
        _byte_encode,
        _special_ids(),
        state="[MASK] injected",
        instructions="[MASK] q",
        options=["a", "b"],
        max_len=512,
        head_max_len=192,
    )
    assert ids.count(3) == 2 and len(markers) == 2


def test_laya_temperature_bucket_and_confidence() -> None:
    from router.decide import laya_confidence, laya_temperature

    config = {
        "temperature": [1.5, 1.2, 1.9],
        "temperature_by_options": {"choice:2": 1.9, "choice:3-5": 1.7},
    }
    assert laya_temperature(config, 2) == 1.9
    assert laya_temperature(config, 4) == 1.7
    assert laya_temperature(config, 12) == 1.5
    assert laya_temperature({}, 2) == 1.0
    assert laya_confidence([0.5, 0.5]) == pytest.approx(0.0)
    assert laya_confidence([1.0, 0.0]) == pytest.approx(1.0)
    assert laya_confidence([1.0]) == 1.0


# --- OnnxLayaScorer: full wiring exercised against fake onnxruntime,
# numpy and tokenizers --------------------------------------------------


class _FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _FakeTokenizer:
    @staticmethod
    def from_file(path: str) -> _FakeTokenizer:
        assert path.endswith("tokenizer.json")
        return _FakeTokenizer()

    def encode(self, text: str, add_special_tokens: bool = True) -> _FakeEncoding:
        assert add_special_tokens is False
        return _FakeEncoding(_byte_encode(text))

    def token_to_id(self, token: str) -> int | None:
        return _SPECIAL.get(token)


class _FakeTokenizers:
    Tokenizer = _FakeTokenizer


class _FakeNumpy:
    int64 = "int64"
    bool_ = "bool"

    @staticmethod
    def array(value: Any, dtype: Any = None) -> Any:
        return value


class _FakeSession:
    seen: dict[str, Any] = {}

    def __init__(self, path: str, providers: list[str]) -> None:
        self.path = path
        self.providers = providers

    def run(self, output_names: list[str], feed: dict[str, Any]) -> list[Any]:
        assert output_names == ["logits", "act_probs"]
        assert set(feed) == {
            "input_ids",
            "attention_mask",
            "marker_pos",
            "marker_mask",
            "qtype",
        }
        _FakeSession.seen = feed
        k = len(feed["marker_pos"][0])
        # Index 1 wins whatever the option count.
        logits = [[0.1 if i != 1 else 5.0 for i in range(k)]]
        return [logits, [[0.3, 0.7]]]


class _FakeOnnxruntime:
    InferenceSession = _FakeSession


def _bundle(tmp_path: Path, config: dict[str, Any] | None = None) -> Path:
    model_path = tmp_path / "laya.onnx"
    model_path.write_bytes(b"fake")
    (tmp_path / "laya_config.json").write_text(
        json.dumps(
            config
            if config is not None
            else {
                "max_len": 512,
                "head_max_len": 192,
                "temperature": [1.6, 1.3, 2.0],
                "temperature_by_options": {"choice:2": 1.9},
            }
        )
    )
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer" / "tokenizer.json").write_text("{}")
    return model_path


def _fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys as _sys

    monkeypatch.setitem(_sys.modules, "onnxruntime", _FakeOnnxruntime())
    monkeypatch.setitem(_sys.modules, "numpy", _FakeNumpy())
    monkeypatch.setitem(_sys.modules, "tokenizers", _FakeTokenizers())


def test_onnx_laya_scorer_full_flow_with_fake_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from router.decide import DEFAULT_TIER_OPTIONS, OnnxLayaScorer

    _fake_runtime(monkeypatch)
    scorer = OnnxLayaScorer(model_path=_bundle(tmp_path))
    decision = scorer.predict("hello world")
    assert decision.tier == list(DEFAULT_TIER_OPTIONS)[1]
    assert 0.0 < decision.confidence <= 1.0
    feed = _FakeSession.seen
    assert feed["qtype"] == [0]
    assert len(feed["input_ids"][0]) == len(feed["attention_mask"][0])
    assert all(v == 1 for v in feed["attention_mask"][0])
    assert feed["marker_mask"] == [[True, True]]
    assert all(feed["input_ids"][0][m] == 3 for m in feed["marker_pos"][0])


def test_onnx_laya_scorer_custom_options_and_abstain_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from router.decide import OnnxLayaScorer

    _fake_runtime(monkeypatch)
    scorer = OnnxLayaScorer(
        model_path=_bundle(tmp_path),
        options={"x": "first", "y": "second", "z": "third"},
    )
    assert scorer.predict("p").tier == "y"
    (tmp_path / "b").mkdir()
    strict = OnnxLayaScorer(
        model_path=_bundle(tmp_path / "b"),
        options={"x": "first", "y": "second", "z": "third"},
        abstain_below=1.01,
    )
    assert strict.predict("p").tier is None


def test_onnx_laya_scorer_missing_model_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from router.decide import OnnxLayaScorer

    _fake_runtime(monkeypatch)
    with pytest.raises(FileNotFoundError):
        OnnxLayaScorer(model_path=tmp_path / "missing.onnx")


def test_onnx_laya_scorer_missing_bundle_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from router.decide import OnnxLayaScorer

    _fake_runtime(monkeypatch)
    model_path = _bundle(tmp_path)
    (tmp_path / "tokenizer" / "tokenizer.json").unlink()
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        OnnxLayaScorer(model_path=model_path)


def test_onnx_laya_scorer_empty_logits_abstains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys as _sys

    class _EmptySession:
        def __init__(self, path: str, providers: list[str]) -> None:
            pass

        def run(self, output_names: list[str], feed: dict[str, Any]) -> list[Any]:
            return [[], []]

    class _EmptyOnnxruntime:
        InferenceSession = _EmptySession

    _fake_runtime(monkeypatch)
    monkeypatch.setitem(_sys.modules, "onnxruntime", _EmptyOnnxruntime())

    from router.decide import OnnxLayaScorer

    decision = OnnxLayaScorer(model_path=_bundle(tmp_path)).predict("hello")
    assert decision.tier is None
    assert decision.confidence == 0.0


# --- baseline map: model names folded onto the label vocabulary -----------


def test_evaluate_baseline_map_folds_model_names_onto_labels() -> None:
    rows = [
        LabelledRow.from_dict(_row(model_param="sonnet")),
        LabelledRow.from_dict(_row(tool_use_id="tu-2", model_param="opus")),
    ]
    plain = evaluate(rows, FakeScorer(tier="mid-sufficient"))
    assert plain.agreement_with_baseline == 0.0
    mapped = evaluate(
        rows,
        FakeScorer(tier="mid-sufficient"),
        baseline_map={"sonnet": "mid-sufficient", "opus": "needs-large"},
    )
    assert mapped.agreement_with_baseline == 0.5


def test_parse_baseline_map() -> None:
    from router.decide import DEFAULT_BASELINE_MAP, parse_baseline_map

    assert parse_baseline_map("") == {}
    assert parse_baseline_map("a=b, c=d") == {"a": "b", "c": "d"}
    assert parse_baseline_map(None) == DEFAULT_BASELINE_MAP
    with pytest.raises(ValueError):
        parse_baseline_map("nonsense")


def test_main_baseline_map_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import router.decide as decide_module

    labels_path = _write_jsonl(tmp_path / "labels.jsonl", [_row(model_param="sonnet")])
    rc = decide_module.main(
        [
            "--labels",
            str(labels_path),
            "--scorer",
            "fixed:mid-sufficient",
            "--baseline-map",
            "sonnet=mid-sufficient",
            "--json",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["agreement_with_baseline"] == 1.0


def test_main_dump_writes_one_line_per_scored_row(tmp_path: Path) -> None:
    import router.decide as decide_module

    labels_path = _write_jsonl(
        tmp_path / "labels.jsonl",
        [_row(tier_label="mid-sufficient"), _row(tool_use_id="tu-2", prompt=None)],
    )
    dump = tmp_path / "dump.jsonl"
    rc = decide_module.main(
        [
            "--labels",
            str(labels_path),
            "--scorer",
            "fixed:mid-sufficient",
            "--dump",
            str(dump),
            "--json",
        ]
    )
    assert rc == 0
    lines = [json.loads(line) for line in dump.read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["tool_use_id"] == "tu-1"
    assert lines[0]["tier"] == "mid-sufficient"
    assert lines[0]["tier_label"] == "mid-sufficient"
    assert lines[0]["baseline"] == "sonnet"
    assert lines[0]["latency_s"] >= 0.0
