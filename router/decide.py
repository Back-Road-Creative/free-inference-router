"""Offline evaluation harness for a candidate tier-decision scorer (Laya or
otherwise) against a labelled dispatch-outcome JSONL.

JP decision D1 (2026-09-22): onnxruntime and the Laya weights are approved
for **offline evaluation only**. Laya is a 421M-param non-autoregressive
typed-decision model with an ONNX Runtime port. This module must NEVER be
wired into live selection, permission, or release decisions --
``tests/test_decide_not_wired.py`` asserts ``router.selector`` (and every
other module under ``router/``) never imports this one.

The router's ``pyproject.toml`` keeps ``dependencies = []`` deliberately (a
security boundary -- see the comment there). onnxruntime is added ONLY as
the optional ``laya`` extra (``pip install -e ".[laya]"``); importing this
module, or using ``FixedScorer``, never requires it. ``OnnxLayaScorer``
imports onnxruntime lazily inside ``__init__`` so the package and the full
test suite work without the extra installed.

No network access anywhere in this module: the ONNX model is always a local
file path supplied by the caller; nothing here downloads weights.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

REQUIRED_SCHEMA = "dispatch-outcome/v1"

#: Chars-per-token approximation (~1,600 chars / 512 tokens, .data/audits/
#: 2026-09-23-laya-offline-eval.md); used by candidates.enforce_token_budget.
CHARS_PER_TOKEN_APPROX = 3.2


def approx_token_count(
    text: str, chars_per_token: float = CHARS_PER_TOKEN_APPROX
) -> int:
    """Char->token approximation, ceil-based so text is never under
    budget by rounding down."""
    if not text:
        return 0
    return math.ceil(len(text) / chars_per_token)


#: Fields every valid row must carry (POLICY.md-adjacent contract for
#: schema "dispatch-outcome/v1" -- rows missing any of these are skipped
#: and counted by ``load_labels``, never raised past the loader).
_REQUIRED_FIELDS = (
    "schema",
    "session",
    "tool_use_id",
    "ts",
    "subagent_type",
    "model_param",
    "background",
    "prompt_sha",
    "prompt_chars",
    "prompt",
    "outcome",
    "redispatched_by",
    "tier_label",
)


@dataclass(frozen=True)
class Decision:
    """A scorer's output for one prompt. ``tier is None`` means abstain."""

    tier: str | None
    confidence: float


class Scorer(Protocol):
    def predict(self, prompt: str) -> Decision: ...


@dataclass(frozen=True)
class FixedScorer:
    """Trivial deterministic scorer for smoke runs (``--scorer fixed:<tier>``).

    Returns the configured tier (or ``None`` to always abstain) for every
    prompt, confidence 1.0. No I/O, no model.
    """

    tier: str | None

    def predict(self, prompt: str) -> Decision:
        return Decision(tier=self.tier, confidence=1.0)


@dataclass(frozen=True)
class LabelledRow:
    """One validated ``dispatch-outcome/v1`` row."""

    schema: str
    session: str
    tool_use_id: str
    ts: str
    subagent_type: str
    model_param: str | None
    background: bool
    prompt_sha: str
    prompt_chars: int
    prompt: str | None
    outcome: str
    redispatched_by: str | None
    tier_label: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabelledRow:
        return cls(
            schema=data["schema"],
            session=data["session"],
            tool_use_id=data["tool_use_id"],
            ts=data["ts"],
            subagent_type=data["subagent_type"],
            model_param=data["model_param"],
            background=data["background"],
            prompt_sha=data["prompt_sha"],
            prompt_chars=data["prompt_chars"],
            prompt=data["prompt"],
            outcome=data["outcome"],
            redispatched_by=data["redispatched_by"],
            tier_label=data["tier_label"],
        )


def load_labels(path: Path) -> tuple[list[LabelledRow], int]:
    """Load and validate a labelled JSONL file.

    Returns ``(rows, bad_count)``. A row is skipped (and counted in
    ``bad_count``, never raised) when: the line is not valid JSON, the
    ``schema`` field isn't exactly ``"dispatch-outcome/v1"``, or any
    required field is missing. Blank lines are silently ignored (not
    counted as bad).
    """
    rows: list[LabelledRow] = []
    bad_count = 0
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                bad_count += 1
                continue
            if not isinstance(data, dict) or data.get("schema") != REQUIRED_SCHEMA:
                bad_count += 1
                continue
            if any(field_name not in data for field_name in _REQUIRED_FIELDS):
                bad_count += 1
                continue
            rows.append(LabelledRow.from_dict(data))
    return rows, bad_count


@dataclass(frozen=True)
class EvalReport:
    n_used: int
    n_excluded_no_prompt: int
    n_bad_rows: int
    abstention_rate: float
    agreement_with_baseline: float | None
    n_labelled: int
    accuracy_vs_label: float | None
    breakdown_by_outcome: dict[str, int]
    predict_latency_p50_s: float
    predict_latency_p95_s: float
    baseline: str = field(default="model_param")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_used": self.n_used,
            "n_excluded_no_prompt": self.n_excluded_no_prompt,
            "n_bad_rows": self.n_bad_rows,
            "abstention_rate": self.abstention_rate,
            "agreement_with_baseline": self.agreement_with_baseline,
            "baseline": self.baseline,
            "n_labelled": self.n_labelled,
            "accuracy_vs_label": self.accuracy_vs_label,
            "breakdown_by_outcome": self.breakdown_by_outcome,
            "predict_latency_p50_s": self.predict_latency_p50_s,
            "predict_latency_p95_s": self.predict_latency_p95_s,
        }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    frac = k - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def evaluate(
    rows: list[LabelledRow],
    scorer: Scorer,
    baseline: str = "model_param",
    bad_rows: int = 0,
    baseline_map: dict[str, str] | None = None,
    on_decision: Callable[[LabelledRow, Decision, float], None] | None = None,
) -> EvalReport:
    """Evaluate ``scorer`` against labelled ``rows``. Pure and offline: no
    network, no write. Rows without a non-empty ``prompt`` are excluded
    (and counted) rather than scored.

    ``baseline_map`` folds baseline values onto the scorer's vocabulary
    before agreement is counted (``model_param`` holds model names such as
    ``sonnet``; a Laya tier question answers in the ``tier_label``
    vocabulary). Unmapped values compare as-is. ``on_decision`` is called
    once per scored row with ``(row, decision, latency_s)`` so a caller can
    dump per-row results without the report growing.
    """
    usable = [r for r in rows if r.prompt]
    n_excluded = len(rows) - len(usable)

    decisions: list[Decision] = []
    latencies: list[float] = []
    for row in usable:
        assert row.prompt is not None
        start = time.perf_counter()
        decision = scorer.predict(row.prompt)
        latency = time.perf_counter() - start
        latencies.append(latency)
        decisions.append(decision)
        if on_decision is not None:
            on_decision(row, decision, latency)

    n_used = len(usable)
    n_abstained = sum(1 for d in decisions if d.tier is None)
    abstention_rate = (n_abstained / n_used) if n_used else 0.0

    mapping = baseline_map or {}
    baseline_values = [
        mapping.get(value, value) if isinstance(value, str) else value
        for value in (getattr(r, baseline) for r in usable)
    ]
    agreement_pairs = [
        (d.tier, b)
        for d, b in zip(decisions, baseline_values, strict=True)
        if b is not None
    ]
    agreement_with_baseline = (
        sum(1 for d_tier, b in agreement_pairs if d_tier == b) / len(agreement_pairs)
        if agreement_pairs
        else None
    )

    labelled_pairs = [
        (d.tier, r.tier_label)
        for d, r in zip(decisions, usable, strict=True)
        if r.tier_label is not None
    ]
    n_labelled = len(labelled_pairs)
    accuracy_vs_label = (
        sum(1 for d_tier, label in labelled_pairs if d_tier == label) / n_labelled
        if n_labelled
        else None
    )

    breakdown_by_outcome = dict(Counter(r.outcome for r in usable))

    p50 = _percentile(latencies, 0.50)
    p95 = _percentile(latencies, 0.95)

    return EvalReport(
        n_used=n_used,
        n_excluded_no_prompt=n_excluded,
        n_bad_rows=bad_rows,
        abstention_rate=abstention_rate,
        agreement_with_baseline=agreement_with_baseline,
        baseline=baseline,
        n_labelled=n_labelled,
        accuracy_vs_label=accuracy_vs_label,
        breakdown_by_outcome=breakdown_by_outcome,
        predict_latency_p50_s=p50,
        predict_latency_p95_s=p95,
    )


#: Laya question types (``QTYPES`` in the checkpoint's rl_common.py and in
#: the receptron/laya export). A tier decision is a ``choice``.
_QTYPE_CHOICE = 0

#: The tier question OnnxLayaScorer asks by default. Option KEYS are the
#: ``tier_label`` vocabulary bin/dispatch-outcomes.py emits, so
#: ``accuracy_vs_label`` compares like with like; the VALUES are the
#: criteria text Laya sees next to each option.
DEFAULT_TIER_INSTRUCTIONS = (
    "Which model tier does this delegated agent task need to finish correctly?"
)
DEFAULT_TIER_OPTIONS: dict[str, str] = {
    "mid-sufficient": (
        "a mid-tier model (Sonnet) completes it correctly on its own; "
        "mechanical, well-specified, or read-only work"
    ),
    "needs-large": (
        "only a frontier model (Opus) gets it right; a mid-tier attempt "
        "fails, is escalated, or needs deep design judgement"
    ),
}

#: ``model_param`` values folded onto that vocabulary for
#: ``agreement_with_baseline`` (the CLI default for ``--baseline-map``).
DEFAULT_BASELINE_MAP: dict[str, str] = {
    "haiku": "mid-sufficient",
    "sonnet": "mid-sufficient",
    "opus": "needs-large",
    "fable": "needs-large",
}


def parse_baseline_map(spec: str | None) -> dict[str, str]:
    """``"a=b, c=d"`` -> ``{"a": "b", "c": "d"}``; ``None`` -> the default
    map; ``""`` -> no mapping. A pair without ``=`` is a ``ValueError``."""
    if spec is None:
        return dict(DEFAULT_BASELINE_MAP)
    mapping: dict[str, str] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"--baseline-map pair without '=': {pair!r}")
        key, value = pair.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


@dataclass(frozen=True)
class LayaSpecialIds:
    cls: int
    sep: int
    mask: int
    pad: int


_MASK_TOKEN = "[MASK]"


def build_laya_sequence(
    encode: Callable[[str], list[int]],
    ids: LayaSpecialIds,
    state: str,
    instructions: str,
    options: list[str],
    max_len: int,
    head_max_len: int,
) -> tuple[list[int], list[int]]:
    """Port of ``rl_common.build_sequence`` for a ``choice`` question::

        [CLS] choice question: <instructions> [SEP]
        [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]

    Returns ``(ids, marker_positions)`` -- the position of each option's
    ``[MASK]`` is where the graph reads that option's logit. The literal
    mask token is scrubbed from user text so a prompt cannot inject a
    marker. The state is truncated to what fits under ``max_len`` after the
    question head; option texts are capped at 48 tokens each and shrunk
    evenly when the head would exceed ``head_max_len``.
    """

    def scrub(text: str) -> str:
        return text.replace(_MASK_TOKEN, " ")

    head = encode(f"choice question: {scrub(instructions)}")
    opt_ids = [[ids.mask, *encode(" " + scrub(o))[:48]] for o in options]
    budget = head_max_len - sum(len(o) for o in opt_ids)
    if budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        budget = head_max_len - sum(len(o) for o in opt_ids)
    head = head[: max(8, budget)]
    seq = [ids.cls, *head, ids.sep]
    markers: list[int] = []
    for o in opt_ids:
        markers.append(len(seq))
        seq.extend(o)
    seq.append(ids.sep)
    room = max(0, max_len - len(seq) - 1)
    seq.extend(encode(scrub(state))[:room])
    seq.append(ids.sep)
    return seq[:max_len], [m for m in markers if m < max_len]


def laya_temperature(config: dict[str, Any], n_options: int) -> float:
    """Per-cardinality softmax temperature from ``laya_config.json``
    (``temperature_by_options["choice:<bucket>"]``, else
    ``temperature[qtype]``, else 1.0)."""
    if n_options <= 2:
        bucket = "2"
    elif n_options <= 5:
        bucket = "3-5"
    elif n_options <= 10:
        bucket = "6-10"
    else:
        bucket = "11+"
    by_options = config.get("temperature_by_options") or {}
    value = by_options.get(f"choice:{bucket}")
    if value is None:
        temps = config.get("temperature") or []
        value = temps[_QTYPE_CHOICE] if len(temps) > _QTYPE_CHOICE else 1.0
    return float(value)


def laya_confidence(probs: list[float]) -> float:
    """Jev-style confidence: 1 minus the normalised entropy of the answer
    distribution (1.0 = certain, 0.0 = uniform)."""
    k = len(probs)
    if k < 2:
        return 1.0
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    return 1.0 - entropy / math.log(k)


def _softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


class OnnxLayaScorer:
    """Runs the Laya ONNX Runtime export against a local bundle.

    OFFLINE EVAL ONLY (JP D1, 2026-09-22) -- see module docstring.
    onnxruntime, numpy and tokenizers are imported lazily, inside
    ``__init__``, so the package and the full test suite work with zero
    third-party dependencies installed; a clear ``RuntimeError`` is raised
    if the ``laya`` extra (``pip install -e ".[laya]"``) is missing.

    ``model_path`` is the bundle's ``laya.onnx``; ``laya_config.json`` and
    ``tokenizer/tokenizer.json`` are read from the same directory
    (``scripts/laya-fetch.sh`` lays the bundle out this way). Nothing is
    downloaded; the caller is responsible for the bundle's provenance.

    The graph (probed 2026-09-23, see README "Getting the weights") takes
    ``input_ids``/``attention_mask`` ``[B, L]``, ``marker_pos``/
    ``marker_mask`` ``[B, K]`` and ``qtype`` ``[B]``, and returns
    ``logits`` ``[B, K]`` (one per option) and ``act_probs`` ``[B, 2]``.
    Each prompt is scored as the STATE of one ``choice`` question whose
    options are the tiers; the answer is the argmax of the
    temperature-scaled softmax, with ``laya_confidence`` as confidence.
    ``abstain_below`` turns a low-confidence answer into an abstention.
    """

    def __init__(
        self,
        model_path: str | Path,
        options: dict[str, str] | None = None,
        instructions: str = DEFAULT_TIER_INSTRUCTIONS,
        abstain_below: float = 0.0,
    ) -> None:
        try:
            import numpy as np  # type: ignore[import-not-found,import-untyped,unused-ignore]
            import onnxruntime as ort  # type: ignore[import-not-found,import-untyped,unused-ignore]
            import tokenizers  # type: ignore[import-not-found,import-untyped,unused-ignore]
        except ImportError as exc:
            raise RuntimeError(
                "OnnxLayaScorer requires the 'laya' extra: "
                'pip install -e ".[laya]" (onnxruntime/tokenizers not installed)'
            ) from exc

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Laya ONNX model not found: {model_path}")
        bundle_dir = model_path.parent
        config_path = bundle_dir / "laya_config.json"
        tokenizer_path = bundle_dir / "tokenizer" / "tokenizer.json"
        for required in (config_path, tokenizer_path):
            if not required.is_file():
                raise FileNotFoundError(f"Laya bundle file not found: {required}")

        self._np: Any = np
        self._config: dict[str, Any] = json.loads(
            config_path.read_text(encoding="utf-8")
        )
        self._max_len = int(self._config.get("max_len", 512))
        self._head_max_len = int(self._config.get("head_max_len", 192))
        self._options = dict(options or DEFAULT_TIER_OPTIONS)
        self._instructions = instructions
        self._abstain_below = abstain_below

        tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        special: dict[str, int] = {}
        for token in ("[CLS]", "[SEP]", "[MASK]", "[PAD]"):
            token_id = tokenizer.token_to_id(token)
            if token_id is None:
                raise ValueError(f"special token {token} missing from {tokenizer_path}")
            special[token] = int(token_id)
        self._ids = LayaSpecialIds(
            cls=special["[CLS]"],
            sep=special["[SEP]"],
            mask=special["[MASK]"],
            pad=special["[PAD]"],
        )
        self._encode: Callable[[str], list[int]] = lambda text: list(
            tokenizer.encode(text, add_special_tokens=False).ids
        )
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

    def _rendered_options(self) -> list[str]:
        return [f"{k}: {v}" if v else k for k, v in self._options.items()]

    def predict(self, prompt: str) -> Decision:
        np = self._np
        ids, markers = build_laya_sequence(
            self._encode,
            self._ids,
            state=prompt,
            instructions=self._instructions,
            options=self._rendered_options(),
            max_len=self._max_len,
            head_max_len=self._head_max_len,
        )
        if len(markers) != len(self._options):
            raise ValueError(
                f"options do not fit in head_max_len={self._head_max_len} tokens"
            )
        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.array([[1] * len(ids)], dtype=np.int64),
            "marker_pos": np.array([markers], dtype=np.int64),
            "marker_mask": np.array([[True] * len(markers)], dtype=np.bool_),
            "qtype": np.array([_QTYPE_CHOICE], dtype=np.int64),
        }
        outputs = self._session.run(["logits", "act_probs"], feed)
        logits = _flatten_to_floats(outputs[0])[: len(markers)]
        if not logits:
            return Decision(tier=None, confidence=0.0)
        temperature = laya_temperature(self._config, len(logits))
        probs = _softmax([x / temperature for x in logits])
        idx = probs.index(max(probs))
        confidence = laya_confidence(probs)
        if confidence < self._abstain_below:
            return Decision(tier=None, confidence=confidence)
        return Decision(tier=list(self._options)[idx], confidence=confidence)


def _flatten_to_floats(value: Any) -> list[float]:
    """Flatten a nested list, tuple, or numpy-array-like (anything with
    ``.tolist()``) into a flat list of floats."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    flat: list[float] = []

    def _walk(v: Any) -> None:
        if isinstance(v, (list, tuple)):
            for item in v:
                _walk(item)
        else:
            flat.append(float(v))

    _walk(value)
    return flat


def _build_scorer(args: argparse.Namespace) -> Scorer:
    if args.scorer and args.scorer.startswith("fixed:"):
        tier = args.scorer.split(":", 1)[1]
        return FixedScorer(tier=tier if tier != "none" else None)
    if args.model:
        return OnnxLayaScorer(model_path=args.model, abstain_below=args.abstain_below)
    raise SystemExit("must supply either --model <onnx-path> or --scorer fixed:<tier>")


#: Approx-token budget for a candidate row's state_text (mirrors the Laya
#: bundle's head_max_len headroom).
_CANDIDATE_STATE_TOKEN_BUDGET = 320


def _run_candidates(args: argparse.Namespace) -> dict[str, Any]:
    """``--candidates`` CLI path (Y0, plan §3.7); never touches
    router.selector."""
    from router import candidates as cand

    rows, bad_counts = cand.load_candidates(args.candidates)
    splits = cand.split(rows, dev_frac=0.6, cal_frac=0.2, seed=0)
    dev = splits["dev"]
    selected = splits[args.split] if args.split else rows

    if args.split == "test":
        cand.ensure_test_split_authorized(args.prereg, Path(cand.__file__))

    class_report = cand.class_count_report(selected)
    classes = sorted({option for row in rows for option in row.options})

    majority = cand.MajorityBaseline(cand.fit_majority(dev))
    rule = cand.RuleBaseline("always_first_option")
    baselines: dict[str, Any] = {
        "majority": cand.score_predictions(majority.predict, selected, classes),
        "always_first_option": cand.score_predictions(rule.predict, selected, classes),
    }

    laya: dict[str, Any] | None = None
    if args.model or (args.scorer and not args.scorer.startswith("fixed:")):
        scorer = _build_scorer(args)
        n_labelled = 0
        n_correct = 0
        n_truncated = 0
        for row in selected:
            if row.label is None:
                continue
            text, truncated = cand.enforce_token_budget(
                cand.state_text(row),
                max_tokens=_CANDIDATE_STATE_TOKEN_BUDGET,
                approx_tokens=approx_token_count,
            )
            n_truncated += int(truncated)
            decision = scorer.predict(text)
            n_labelled += 1
            n_correct += int(decision.tier == row.label)
        laya = {
            "n_labelled": n_labelled,
            "n_truncated": n_truncated,
            "accuracy": (n_correct / n_labelled) if n_labelled else None,
        }

    return {
        "schema": cand.CANDIDATE_SCHEMA,
        "n_rows": len(rows),
        "bad_counts": dict(bad_counts),
        "split": args.split,
        "class_counts": class_report["class_counts"],
        "n_unlabelled": class_report["n_unlabelled"],
        "insufficient_data": not class_report["evaluable"],
        "baselines": baselines,
        "laya": laya,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m router.decide",
        description=(
            "Offline evaluation of a candidate tier-decision scorer against "
            "a labelled dispatch-outcome/v1 JSONL, or (--candidates) the "
            "task-generic laya-candidate/v1 schema. Never wired into live "
            "selection -- offline eval only (JP D1, 2026-09-22)."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--labels", type=Path, default=None)
    source.add_argument("--candidates", type=Path, default=None)
    parser.add_argument(
        "--prereg", type=Path, default=None, help="required to score --split test"
    )
    parser.add_argument(
        "--split",
        choices=["dev", "cal", "test"],
        default=None,
        help="with --candidates: score this grouped split only (default: all rows)",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--scorer", type=str, default=None)
    parser.add_argument("--baseline", type=str, default="model_param")
    parser.add_argument(
        "--baseline-map",
        type=str,
        default=None,
        help=(
            "comma-separated k=v pairs folding baseline values onto the "
            "scorer's vocabulary before agreement is counted; default maps "
            "model names (haiku/sonnet/opus/fable) onto tier labels; '' disables"
        ),
    )
    parser.add_argument(
        "--abstain-below",
        type=float,
        default=0.0,
        help="Laya answers below this confidence count as abstentions",
    )
    parser.add_argument(
        "--dump",
        type=Path,
        default=None,
        help="write one JSON line per scored row (decision, label, baseline) here",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.candidates:
        result = _run_candidates(args)
        if args.json:
            print(json.dumps(result))
        else:
            for key, value in result.items():
                print(f"{key}: {value}")
        return 0

    scorer = _build_scorer(args)
    rows, bad_count = load_labels(args.labels)
    dump_lines: list[str] = []

    def _record(row: LabelledRow, decision: Decision, latency: float) -> None:
        dump_lines.append(
            json.dumps(
                {
                    "tool_use_id": row.tool_use_id,
                    "prompt_sha": row.prompt_sha,
                    "prompt_chars": row.prompt_chars,
                    "tier": decision.tier,
                    "confidence": decision.confidence,
                    "tier_label": row.tier_label,
                    "baseline": getattr(row, args.baseline),
                    "outcome": row.outcome,
                    "latency_s": latency,
                }
            )
        )

    report = evaluate(
        rows,
        scorer,
        baseline=args.baseline,
        bad_rows=bad_count,
        baseline_map=parse_baseline_map(args.baseline_map),
        on_decision=_record if args.dump else None,
    )
    if args.dump:
        args.dump.write_text("".join(line + "\n" for line in dump_lines))

    if args.json:
        print(json.dumps(report.to_dict()))
    else:
        for key, value in report.to_dict().items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess CLI test
    sys.exit(main())
