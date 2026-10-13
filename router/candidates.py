"""Task-generic offline ``laya-candidate/v1`` schema (plan §3.7, unit Y0).
Stdlib-only; refuses a held-out test split unless the feature builder was
frozen first (``ensure_test_split_authorized``). Not wired into live
selection (``tests/test_decide_not_wired``)."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CANDIDATE_SCHEMA = "laya-candidate/v1"

#: Every valid row must carry all of these; each gets its own error code.
_REQUIRED_CANDIDATE_FIELDS = (
    "schema",
    "task_type",
    "candidate_id",
    "ts",
    "group_id",
    "features",
    "options",
    "label",
    "label_provenance",
    "censored",
    "outcome",
)

#: Strongest to weakest label confidence.
VALID_PROVENANCE = ("adjudicated", "outcome", "rerun", "unknown")

#: A class needs at least this many labelled rows to be evaluable.
CLASS_FLOOR = 20

_MAX_FEATURE_STRING_LEN = 200


def validate_candidate_dict(data: Any) -> str | None:
    """Validate one JSON-decoded row; error code or ``None``, never raises."""
    if not isinstance(data, dict):
        return "not_object"
    if data.get("schema") != CANDIDATE_SCHEMA:
        return "wrong_schema"
    missing = [f for f in _REQUIRED_CANDIDATE_FIELDS if f not in data]
    if missing:
        return f"missing_field:{missing[0]}"

    if not isinstance(data["task_type"], str) or not data["task_type"]:
        return "bad_task_type"
    if not isinstance(data["candidate_id"], str) or not data["candidate_id"]:
        return "bad_candidate_id"
    if not isinstance(data["group_id"], str) or not data["group_id"]:
        return "bad_group_id"
    if not isinstance(data["ts"], str) or not data["ts"]:
        return "bad_ts"

    features = data["features"]
    if not isinstance(features, dict):
        return "bad_features"
    for value in features.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, str):
            if len(value) > _MAX_FEATURE_STRING_LEN:
                return "bad_features"
        elif not isinstance(value, (int, float)) and value is not None:
            return "bad_features"

    options = data["options"]
    if (
        not isinstance(options, list)
        or not options
        or not all(isinstance(o, str) and o for o in options)
        or len(set(options)) != len(options)
    ):
        return "bad_options"

    label = data["label"]
    if label is not None and (not isinstance(label, str) or label not in options):
        return "bad_label"

    if data["label_provenance"] not in VALID_PROVENANCE:
        return "bad_label_provenance"

    if not isinstance(data["censored"], bool):
        return "bad_censored"

    outcome = data["outcome"]
    if outcome is not None and not isinstance(outcome, str):
        return "bad_outcome"

    return None


@dataclass(frozen=True)
class CandidateRow:
    """One validated ``laya-candidate/v1`` row."""

    schema: str
    task_type: str
    candidate_id: str
    ts: str
    group_id: str
    features: dict[str, Any]
    options: list[str]
    label: str | None
    label_provenance: str
    censored: bool
    outcome: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateRow:
        return cls(
            schema=data["schema"],
            task_type=data["task_type"],
            candidate_id=data["candidate_id"],
            ts=data["ts"],
            group_id=data["group_id"],
            features=dict(data["features"]),
            options=list(data["options"]),
            label=data["label"],
            label_provenance=data["label_provenance"],
            censored=data["censored"],
            outcome=data["outcome"],
        )


def load_candidates(path: Path) -> tuple[list[CandidateRow], Counter[str]]:
    """Returns ``(rows, bad_counts)``; a bad row is counted, never dropped
    silently. Blank lines are ignored."""
    rows: list[CandidateRow] = []
    bad_counts: Counter[str] = Counter()
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                bad_counts["bad_json"] += 1
                continue
            error = validate_candidate_dict(data)
            if error is not None:
                bad_counts[error] += 1
                continue
            rows.append(CandidateRow.from_dict(data))
    return rows, bad_counts


def split(
    rows: list[CandidateRow], dev_frac: float, cal_frac: float, seed: int
) -> dict[str, list[CandidateRow]]:
    """Grouped, time-ordered split; no ``group_id`` on two sides. Groups
    order by earliest ``ts`` (``seed`` breaks ties only); ``dev`` holds
    the earliest groups, ``test`` the latest."""
    if dev_frac < 0 or cal_frac < 0 or dev_frac + cal_frac > 1:
        raise ValueError("dev_frac/cal_frac must be >= 0 and sum to <= 1")

    groups: dict[str, list[CandidateRow]] = {}
    for row in rows:
        groups.setdefault(row.group_id, []).append(row)

    gids = list(groups)
    random.Random(seed).shuffle(gids)
    ordered = sorted(gids, key=lambda gid: min(r.ts for r in groups[gid]))

    n = len(ordered)
    n_dev = round(n * dev_frac)
    n_cal = round(n * cal_frac)
    dev_gids = set(ordered[:n_dev])
    cal_gids = set(ordered[n_dev : n_dev + n_cal])

    out: dict[str, list[CandidateRow]] = {"dev": [], "cal": [], "test": []}
    for row in rows:
        if row.group_id in dev_gids:
            out["dev"].append(row)
        elif row.group_id in cal_gids:
            out["cal"].append(row)
        else:
            out["test"].append(row)
    return out


def class_count_report(rows: list[CandidateRow]) -> dict[str, Any]:
    """Per-class counts plus ``evaluable``: >=2 classes each >= floor."""
    counts = Counter(row.label for row in rows if row.label is not None)
    n_unlabelled = sum(1 for row in rows if row.label is None)
    evaluable = len(counts) >= 2 and all(c >= CLASS_FLOOR for c in counts.values())
    return {
        "class_counts": dict(counts),
        "n_unlabelled": n_unlabelled,
        "evaluable": evaluable,
        "floor": CLASS_FLOOR,
    }


_REQUIRED_PREREG_CANDIDATE_FIELDS = (
    "metrics",
    "false_negative_limit",
    "coverage_floor",
    "latency_budget_s",
)


class PreregError(Exception):
    """Prereg missing/malformed, or the frozen builder hash no longer
    matches -- refuses to score a held-out test split."""


def feature_builder_sha256(builder_path: Path) -> str:
    """``sha256:<hex>`` of the frozen feature-builder module's source. The
    ``sha256:`` prefix marks the value as a content hash, so a secrets
    scanner never reads the prereg file's copy as a bare hex credential."""
    return "sha256:" + hashlib.sha256(Path(builder_path).read_bytes()).hexdigest()


def load_prereg(path: Path) -> dict[str, Any]:
    """Raises ``PreregError`` on a missing hash or missing candidate
    fields (see ``_REQUIRED_PREREG_CANDIDATE_FIELDS``)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise PreregError("prereg top level is not an object")
    data: dict[str, Any] = raw
    if "feature_builder_sha256" not in data:
        raise PreregError("prereg missing feature_builder_sha256")
    candidates = data.get("candidates")
    if not isinstance(candidates, dict) or not candidates:
        raise PreregError("prereg missing non-empty 'candidates'")
    for candidate_id, spec in candidates.items():
        if not isinstance(spec, dict):
            raise PreregError(f"prereg candidate {candidate_id!r} is not an object")
        missing = [f for f in _REQUIRED_PREREG_CANDIDATE_FIELDS if f not in spec]
        if missing:
            raise PreregError(f"prereg candidate {candidate_id!r} missing: {missing}")
    return data


def ensure_test_split_authorized(
    prereg_path: Path | None, builder_path: Path
) -> dict[str, Any]:
    """Raises ``PreregError`` unless ``prereg_path`` exists and its hash
    matches ``builder_path``'s current content."""
    if prereg_path is None or not Path(prereg_path).is_file():
        raise PreregError("prereg file required before scoring a test split")
    data = load_prereg(prereg_path)
    current = feature_builder_sha256(builder_path)
    recorded = data["feature_builder_sha256"]
    if recorded != current:
        raise PreregError(
            "feature-builder hash mismatch: freeze the builder before "
            f"test-split scoring (recorded {recorded!r}, current {current!r})"
        )
    return data


def state_text(row: CandidateRow) -> str:
    """``"k1=v1; k2=v2; ..."`` over sorted keys -- the frozen feature
    builder whose sha256 a prereg records."""
    parts = [f"{k}={v}" for k, v in sorted(row.features.items(), key=lambda kv: kv[0])]
    return "; ".join(parts)


def enforce_token_budget(
    text: str, max_tokens: int, approx_tokens: Callable[[str], int]
) -> tuple[str, bool]:
    """``(text, truncated)`` -- truncates to fit ``max_tokens`` per
    ``approx_tokens``; truncation is always flagged, never silent."""
    if approx_tokens(text) <= max_tokens:
        return text, False
    ratio = max_tokens / approx_tokens(text)
    cut = max(1, int(len(text) * ratio))
    truncated_text = text[:cut]
    while approx_tokens(truncated_text) > max_tokens and len(truncated_text) > 1:
        truncated_text = truncated_text[:-1]
    return truncated_text, True


def fit_majority(rows: list[CandidateRow]) -> str | None:
    """Most common label in ``rows`` (fit on dev); ``None`` if unlabelled."""
    labelled = [row.label for row in rows if row.label is not None]
    if not labelled:
        return None
    return Counter(labelled).most_common(1)[0][0]


@dataclass(frozen=True)
class MajorityBaseline:
    label: str | None

    def predict(self, row: CandidateRow) -> str | None:
        return self.label


def always_first_option(row: CandidateRow) -> str | None:
    """The only built-in rule: always the first fixed-order option."""
    return row.options[0] if row.options else None


#: name -> callable; a future rule is added here.
RULE_REGISTRY: dict[str, Callable[[CandidateRow], str | None]] = {
    "always_first_option": always_first_option,
}


@dataclass(frozen=True)
class RuleBaseline:
    name: str

    def predict(self, row: CandidateRow) -> str | None:
        return RULE_REGISTRY[self.name](row)


def score_predictions(
    predict_fn: Callable[[CandidateRow], str | None],
    rows: list[CandidateRow],
    classes: list[str],
) -> dict[str, Any]:
    """Confusion matrix, per-class precision/recall, balanced accuracy,
    and coverage for ``predict_fn`` over labelled ``rows``."""
    labelled = [row for row in rows if row.label is not None]
    n = len(labelled)
    pairs = [(row.label, predict_fn(row)) for row in labelled]
    n_abstained = sum(1 for _, pred in pairs if pred is None)
    coverage = (n - n_abstained) / n if n else 0.0

    confusion: dict[str, dict[str, int]] = {
        cls: dict.fromkeys(classes, 0) for cls in classes
    }
    for true, pred in pairs:
        if pred is not None and true in confusion and pred in confusion[true]:
            confusion[true][pred] += 1

    per_class: dict[str, dict[str, float]] = {}
    recalls: list[float] = []
    for cls in classes:
        tp = confusion[cls][cls]
        fn = sum(confusion[cls].values()) - tp
        fp = sum(confusion[other][cls] for other in classes) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_class[cls] = {"precision": precision, "recall": recall}
        recalls.append(recall)
    balanced_accuracy = sum(recalls) / len(recalls) if recalls else 0.0

    return {
        "n": n,
        "n_abstained": n_abstained,
        "coverage": coverage,
        "confusion": confusion,
        "per_class": per_class,
        "balanced_accuracy": balanced_accuracy,
    }
