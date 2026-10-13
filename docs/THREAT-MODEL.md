# Threat Model — Phase 0a

No runtime exists yet. Belt (robust by construction) / suspenders (upstream
gate) any future runtime must close before admission, per `docs/POLICY.md`.

## Prompt/code retention and training use

- Belt: only `no-training`-class providers ever receive non-synthetic data
  (D1, D4); Gemini receives synthetic-benchmark data only.
- Suspenders: dispatcher checks `data_policy_class` against the payload's
  data class before every call; `unknown` fails closed as `training`.

## Provider/model substitution and poisoned output

- Belt: response `model_id` is compared to the request (mismatch hard-stops
  and quarantines the provider); remote output is treated as **data**,
  never commands — the judge never executes text found in a response.
- Suspenders: a mid-task model switch starts a fresh attempt in a clean
  disposable worktree, never a resumed session; tool calls go through
  job-catalog/argv rendering (`policy.job_argv()`) — the model selects a
  job id, never a shell string.

## Tool permissions and log content

- Belt: R1 tasks are read/diagnose/small-diff shaped only (POLICY.md); logs
  capture the outbound archive's file list and classification decision,
  never raw provider responses above `synthetic`.
- Suspenders: `allowed_roles` per registry entry is enforced before
  dispatch, not left to prompt instructions; `detect-secrets` scans the
  archive before send and a hit hard-stops before any log entry is written.

## Cost escape and credential exposure

- Belt: zero-spend rule — non-zero or unknown price hard-stops; admitted
  accounts carry no payment method or balance; provider keys, once they
  exist, live only in `sops-edit`-managed secrets JP handles directly (D6).
- Suspenders: per-provider RPM/RPD/TPM/TPD counters checked before dispatch,
  never discovered via a 429; `$HOME/.codex/auth.json` and `$HOME/.claude`
  must never appear in any bind or mount set this service constructs
  (mirrors the local-agent lane's `CODEX_HOME` isolation), tested the way
  `tests/test_confine.py` asserts that boundary, not intent.
