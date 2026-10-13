# svc-free-inference-router

## Purpose

A loopback-only gateway that will route bounded, non-sensitive tasks to
genuinely free-tier remote LLM providers, alongside — never replacing —
a sealed local-agent lane.

## Owner

Back Road Creative.

## Status

**Phase 1 — offline runtime under construction.** `router/` is a dependency-free
Python package (`pyproject.toml`, `tests/`); its policy data and docs: `docs/POLICY.md` (lanes, zero-spend rule, budgets,
data classification, retry rules, kill switch), `docs/THREAT-MODEL.md`
(belt-and-suspenders per threat), and `router/` (registry schema,
`providers.json`, default-deny
`classification.json`). The admitted set is pinned by
`tests/test_package.py` and the repo-root `tests/test_router_registry.py`, and
follows the 2026-09-05 admission-bar evaluation
(`.data/plans/2026-09-05-admission-bar-findings.md`). Phase 1 (the
offline gateway runtime) is underway; JP decoupled it from the usage gate on
2026-08-30 because the baseline turned out to be retrospective. Full context
lives in `.data/plans/2026-08-29-local-first-free-inference-routing.md`. Per D6,
only JP creates provider accounts, accepts terms, and handles keys (via
`sops-edit`) — no agent here ever signs up or touches a key.

## Local provider (bench lane)

The bench runner also knows a `local` provider: a model id prefixed
`local/` (e.g. `local/qwen3.5:4b`, the local-agent lane's model) dials a
loopback Ollama at `http://127.0.0.1:11434/v1` directly — prefix stripped
for the wire name, no gateway, no key, and a hard refusal if the local
base URL is ever anything but loopback. The runner sends
`reasoning_effort: none` on that lane — otherwise qwen3.5 spends the whole
`max_tokens` budget thinking and returns empty content. Its registry row
was admitted 2026-09-01 after a 3/3 live smoke (tool call, long-context
needle, docs task), benchmark/read-only roles; all caps are null (the
quota ledger skips null caps — the local lane is bounded by hardware,
not a provider quota).

## Offline scorer evaluation (`router/decide.py`)

`python -m router.decide --labels <jsonl> --model <onnx> [--json]` scores a
candidate tier-decision model against a labelled `dispatch-outcome/v1`
JSONL (session, prompt, dispatched tier, and outcome) and reports
abstention rate, agreement with the baseline dispatch, accuracy against any
`tier_label`, an outcome breakdown, and predict-latency percentiles. A
`--scorer fixed:<tier>` flag runs a trivial deterministic scorer for smoke
tests without a model. `--dump <jsonl>` writes one line per scored row
(decision, confidence, label, baseline, latency) for a confusion matrix;
`--abstain-below <conf>` turns low-confidence answers into abstentions;
`--baseline-map k=v,…` folds baseline values onto the scorer's vocabulary
before agreement is counted (default: `haiku`/`sonnet` → `mid-sufficient`,
`opus`/`fable` → `needs-large`, the `tier_label` vocabulary
`bin/dispatch-outcomes.py` emits; `--baseline-map ""` disables).

## Task-generic candidate schema (`laya-candidate/v1`, `router/candidates.py`)

Plan §3.7 unit Y0: `python -m router.decide --candidates <jsonl>
[--split dev|cal|test] [--prereg docs/laya-prereg.json]
[--model <onnx> | --scorer fixed:<label>] [--json]` (mutually exclusive
with `--labels`) evaluates any typed-choice task -- grouped time-ordered
split, label provenance, preregistered gates. Details:
`docs/LAYA-CANDIDATES.md`; label-data screen for the three Y1 candidates:
`.data/audits/2026-09-25-y0-data-screen.md`.

This is **offline evaluation only** (JP decision D1, 2026-09-22) — it has
no live seat. It is never wired into `router.selector` or any selection,
permission, or release decision (`tests/test_decide_not_wired.py` asserts
the import boundary). Its only model target today is Laya, a 421M-param
non-autoregressive typed-decision model with an ONNX Runtime port; running
`OnnxLayaScorer` needs the optional `laya` extra
(`pip install -e ".[laya]"`: `onnxruntime>=1.16.0`, `tokenizers>=0.15`) —
importing `router.decide` and running its full test suite never require
it (the suite fakes all three modules), since the router's
`dependencies = []` security boundary stays intact. The model path is
always a local file argument; nothing here downloads weights.

### Getting the weights (offline eval only)

`bash scripts/laya-fetch.sh` (run from this directory) downloads the ONNX
bundle from `receptron/laya-onnx` on huggingface.co — an ONNX Runtime
export of `convaiinnovations/laya` (weights Apache-2.0, export MIT) — into
`${XDG_CACHE_HOME:-~/.cache}/laya/` (`LAYA_DIR` overrides),
records each file's sha256 in `MANIFEST.txt` there, installs the
`laya` extra into `./.venv`, and probes the graph. It is idempotent: a file
whose sha256 already matches the manifest is not re-fetched. The bundle is
five files, about 1.7 GB: `laya.onnx`, `laya.onnx.data`, `laya_config.json`
(`max_len` 512, `head_max_len` 192, per-cardinality temperatures) and
`tokenizer/{tokenizer,tokenizer_config}.json`. Only huggingface.co is
contacted, and only by this script; `router/decide.py` still never
downloads anything. Loading the session takes about 90 s on this box.

The probe on 2026-09-23 reported the real graph, which `OnnxLayaScorer`
drives:

| Tensor | Shape | Type |
|---|---|---|
| in `input_ids` | `[batch, seq]` | int64 |
| in `attention_mask` | `[batch, seq]` | int64 |
| in `marker_pos` | `[batch, options]` | int64 |
| in `marker_mask` | `[batch, options]` | bool |
| in `qtype` | `[batch]` | int64 (0 choice, 1 score, 2 noul) |
| out `logits` | `[batch, options]` | float32, uncalibrated; masked slots −1e4 |
| out `act_probs` | `[batch, 2]` | float32 |

Laya answers a typed question about a state, not a bare prompt. The scorer
asks one `choice` question per prompt — `DEFAULT_TIER_INSTRUCTIONS` with
`DEFAULT_TIER_OPTIONS` (`mid-sufficient` / `needs-large`, each with a
criteria sentence) — and feeds the prompt as the state:
`[CLS] choice question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 [SEP]
<state> [SEP]` (`build_laya_sequence`, a port of the checkpoint's
`rl_common.build_sequence`; the literal `[MASK]` is scrubbed from user text
so a prompt can't inject a marker). `marker_pos` is each option's `[MASK]`
position; the option's logit is read there, scaled by the bundle's
per-cardinality temperature (`laya_temperature`), softmaxed, and the argmax
is the tier with `laya_confidence` (1 − normalised entropy) as confidence.
The state is truncated to what fits under `max_len` (512 tokens) after the
question head, so a long dispatch prompt is judged on its opening.

## Lint

`ruff.toml` extends `pyproject.toml`'s `[tool.ruff.lint]` select (E F I B UP)
with rule families already measured clean for this service (currently W,
TD, ERA) so they can't regress silently. To add another family: fix every
finding `ruff check . --select <FAMILY>` reports first, confirm a clean
run, then add it to `extend-select` — never a per-file-ignore or baseline
to force a dirty family in.

## Exit Condition

If Phase 1's usage-baseline week (D5) never happens, or the plan is
abandoned, this stays a policy-only directory indefinitely — see
`LIFECYCLE.md` for the current state and next review date.

## Exec Sandbox (`router/bench/sandbox.py`)

Exec-scorers (mutation kill, red-test repair) may only run model-generated
code through `run_snippet()`, which wraps each run in a disposable
`docker run` with all of: image pinned by tag **and** digest
(`SANDBOX_IMAGE`), `--network none`, a read-only rootfs with a tmpfs
`/work` workdir, `--cpus`/`--memory`/`--pids-limit` bounds, a hard
time limit, and no host env passthrough. Inputs mount read-only at
`/src`. When docker is unusable it raises `SandboxUnavailableError` —
callers refuse to score, never pass.

The time limit (`timeout_s`, default 30 s) runs **inside** the container
as `timeout -k 5 <s> python …`, so docker's own start-up never counts
against the snippet: on a loaded host start-up alone has run past 30 s,
and a correct snippet used to be scored as a timeout. Exit 124 (TERM at
the limit stopped it) is the only result reported as `timed_out`. A
snippet that ignores TERM gets KILL 5 s later and ends with 137, the same
code an out-of-memory kill gives; the result reports it as `killed`, not
`timed_out`. Either way the sandbox stopped the snippet — it did not fail
on its own. A snippet that itself exits 124 or 137 reads the same way,
which only ever counts against it. The host keeps a backstop of the limit
plus 5 s plus a 120 s docker allowance (`SANDBOX_STARTUP_ALLOWANCE_S`). If
that fires, docker itself stalled: the container is force-removed
(`docker rm -f`, which also clears one created but never started) and the
run raises `SandboxUnavailableError` instead of reporting a timeout. If
the removal fails too, the error says so and the run still refuses.
The suite's no-network proof (`tests/test_bench_sandbox_exec.py`) skips
where docker is absent; a skipped proof is not a pass — it must run green
on the machine where scorers execute before any model output runs there.
The same file's docker-free tests fake `subprocess.run` and pin the
commands themselves (every bound above, the one read-only mount, the
in-container limit, how 124 and 137 read, the removal and refusal when
docker stalls, the refusal when the daemon is unreachable), so a runner with
no docker, such as the CI canary, still checks what is asked for. They do
not prove the kernel enforces it; the docker proofs remain the authority.

Two exec-scorers sit on top of it in `router/bench/scoring.py`:

- **`mutation_kill`** (test-gen role): the reply is generated test code.
  It runs against the pristine module and the planted mutant
  (`pristine_module` / `mutant_module` args, imported as `module_name`,
  default `target`). Pass only if it is green on pristine AND fails on its
  own on the mutant — a run stopped by the time limit or by KILL
  (`timed_out` or `killed`) is not a kill. Identical pristine/mutant
  source is a task bug and raises.
- **`repair_green`** (repair role): the reply is the repaired module. The
  task's own red test (`test_code` arg — required, absent raises) runs
  against it; pass only if it goes green. Nothing from the reply is ever
  used as the test, and a reply carrying `def test_` definitions fails
  outright — the model does not get to modify the judge.

Both raise `SandboxUnavailableError` when docker is unusable — they
refuse to score rather than pass or skip. The bench runner records such a
run with an error starting `sandbox:` and treats it like a 429: the
triple is not done, so the next resume re-runs it instead of keeping a
fail nobody judged. Any other scorer exception is a scorer bug: recorded
as `scorer:…` and done.
