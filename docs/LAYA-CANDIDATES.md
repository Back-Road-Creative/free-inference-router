# Laya candidate schema (`laya-candidate/v1`) -- Y0

Plan §3.7, unit Y0: task-generic offline candidate schema, required
plumbing before any Y1 candidate (CI failure triage, affected-selection
advisory, wake prioritization) is screened. **Offline evaluation only**,
same as `router/decide.py`'s `dispatch-outcome/v1` harness -- never wired
into `router.selector` or any live decision (`tests/test_decide_not_wired.py`
covers this module too).

## Schema

One JSON object per line:

```json
{
  "schema": "laya-candidate/v1",
  "task_type": "ci-failure-triage",
  "candidate_id": "run-12345",
  "ts": "2026-09-24T18:00:00Z",
  "group_id": "pr-987",
  "features": {"n_files_changed": 3, "signature": "assert_fail", "workflow": "CI"},
  "options": ["infrastructure", "regression"],
  "label": "regression",
  "label_provenance": "adjudicated",
  "censored": false,
  "outcome": "confirmed_regression"
}
```

`task_type` names the Y1 candidate; `features` is scalar/short-string only;
`options` is the finite fixed-order label set; `label` is `None` when
unlabelled/censored; `label_provenance` ranks confidence (`adjudicated` >
`outcome` > `rerun` > `unknown`). A bad row is counted, never dropped
silently (`load_candidates` returns `(rows, bad_counts)`).

## Grouped, time-ordered split and class floor

`split(rows, dev_frac, cal_frac, seed)` orders groups by earliest `ts`
(seed only breaks ties) and slices `dev`/`cal`/`test` by fraction -- no
`group_id` ever appears on two sides. `class_count_report(rows)` reports
per-class counts and `evaluable`: at least two classes, each with
`>= CLASS_FLOOR` (20) labelled rows. Below that is **INSUFFICIENT DATA** --
a data-collection gap, never a model failure.

## Preregistration and the frozen builder

`docs/laya-prereg.json` fixes, before any held-out test read, each
candidate's gates plus `feature_builder_sha256` (`sha256:<hex>` of
`router/candidates.py` when `state_text` was frozen).
`ensure_test_split_authorized` raises `PreregError` on a missing/
mismatched hash -- `--split test` calls this automatically.

## Baselines and Laya scoring

Offline, stdlib only: **majority** (`fit_majority`, fit on `dev`) and a
**rule registry** (`RULE_REGISTRY`, only `always_first_option` built in).
`score_predictions` reports a confusion matrix, per-class
precision/recall, balanced accuracy, and coverage. A tiny naive-Bayes
baseline was cut to stay under the ≤1000-line diff cap (Y0's stated
fallback); follow-up: a `predict(row) -> str | None` in the same seam.

A row's `features` become the scorer's prompt via `state_text` (sorted
`"k=v; k=v"`, the frozen builder hashed above). `enforce_token_budget`
truncates against `decide.py`'s `approx_token_count`, always flagging
`truncated=True` -- never silently.

## CLI

```
python -m router.decide --candidates <jsonl> [--split dev|cal|test] [--prereg <json>] [--model <onnx> | --scorer fixed:<label>] [--json]
```

Mutually exclusive with `--labels`. `--json` output: `schema`, `n_rows`,
`bad_counts`, `split`, `class_counts`, `n_unlabelled`, `insufficient_data`,
`baselines`, `laya` (`None` without `--model`/a non-fixed `--scorer`).
`--split test` without a matching `--prereg` exits non-zero.

## How a future evaluation runs

Collect rows for one Y1 candidate (owner: `.data/audits/2026-09-25-y0-data-screen.md`)
until `class_count_report` is `evaluable: true`; freeze `state_text` and record its
hash in `docs/laya-prereg.json` before any `test` read; fit/calibrate on
`--split dev|cal`; score `--split test --prereg docs/laya-prereg.json --model <onnx>`
exactly once; compare against the prereg'd gates -- beating the baselines on
held-out is required, a Y1 candidate winning is not. Stays offline-eval-only until
JP makes a separate, explicit decision to wire a result into a live path.
