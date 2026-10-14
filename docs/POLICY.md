# Provider, Data, and Cost Policy — Phase 0a

Binding for any future runtime here. Decisions D1–D6 are JP 2026-08-29
(`.data/plans/2026-08-29-local-first-free-inference-routing.md`).

## Lanes

| Lane | What | Model? |
|---|---|---|
| D0 | tests, lint, search, codemods | none |
| L1 | sealed local agent (qwen3.5:4b, bubblewrap + loopback netns) | local, offline |
| R1 | free remote provider, per this policy | remote, free tier only |
| C1 | paid subscription client / human | remote, paid, supervised |

Task-shape → lane: deterministic → D0; repo code needing judgment (any
privacy class) → L1; bounded public/synthetic diagnosis or small-diff
review → R1; private, full-repo-write, or over-budget → C1.

## Zero-spend rule and budgets

Dispatch only if a registry entry's `price_usd_per_mtok_in`/`_out` are both
present and **exactly `0`**; missing/null/`>0` hard-stops. Admitted accounts
carry no payment method and no balance. `card_required: unverified` is
treated as `yes` for gating until JP confirms otherwise. RPM/RPD/TPM/TPD
caps live in `router/providers.json`; a remote row with a null `rpd` is
**refused outright** (`uncapped`) — only the `local` lane runs uncapped. Google's
free tier is billing-gated, so its rows carry `billing_gated: true` and are
refused by the selector and the bench runner until `billing_disabled_attested`
(`"<date> JP: <project-id>"`) is set. The attestation names JP's cloud project,
so it lives in the private overlay `$FIR_INPUTS_DIR/providers.local.json`
(merged per provider+model, attest-only), never in the tracked registry, which
ships it null. Re-admitted 2026-09-02 with the AI Studio free-tier caps (flash
5 rpm / 250k tpm / 20 rpd; flash-lite 15 / 250k / 500; reset daily 00:00
Pacific) once JP attested
the attested project (private registry) has no billing account linked. A local counter checks headroom
**before** the call — a 429 never discovers a cap. Groq's binding limit is
TPM (8,000 on the free `gpt-oss` models), so Groq is bounded-artifact roles
only (diagnosis, one test idea, small-diff review) — never whole-file repair.
The registry's `local` row (L1's qwen3.5:4b via loopback Ollama) prices at
0/0 with all caps null — nothing leaves the machine and no quota applies;
the bench runner refuses to dial it anywhere but loopback. Admitted
2026-09-01 after a 3/3 live smoke, benchmark/read-only roles.

Phase-3 admission bars (≥0.80 triage, 100% tool schema, zero resistance leaks,
zero cap violations) were scored against the recorded bench run on 2026-09-05 —
`.data/plans/2026-09-05-admission-bar-findings.md` carries the table and the
command that produces it. Outcomes: `poolside/laguna-s-2.1` quarantined (43 of
50 calls errored, no bar has evidence); `ling-3.0-flash-fin` and
`gemini-3.5-flash` narrowed to `synthetic-benchmark` (triage 0.11 and 0.44);
`gemini-3.5-flash-lite` clears every bar. A row that leaked on the resistance
tasks keeps its roles but carries `output_is_data: true` — its reply may be
consumed by a deterministic judge only, never as another agent's instructions
(groq's two rows and the local row).

## Data classification

Data classes: `public`, `synthetic`, `private`. Provider classes:
`no-training`, `training`, `unknown` (fail-closed as `training`).

| Data class | `no-training` provider | Gemini (`training`) |
|---|---|---|
| public | routable to R1 | not routable |
| synthetic | routable to R1 | routable, benchmark/read-only roles only |
| private | never routable | never routable |

Gemini's free tier is training-class per its unpaid-tier terms and is
admitted only for the synthetic benchmark and public-repo read-only roles
(D4) — no workspace code, public or private, ever reaches it.

## Retry rules

Retry only on: 429, pre-tool-call timeout, transport failure, retryable 5xx.
Hard stop: 401/403, unknown/non-zero price, invalid tool schema, context
overflow, classification denial, or malformed output after a tool has run.
Once any tool has executed, a model switch is a **new attempt from a clean
disposable worktree**, never a resumed session. A returned `model_id`
differing from the request is a hard stop plus quarantine.

## Kill switch, outbound payload, and terms recheck

Checked before every dispatch: `/etc/free-router/disabled`. An **unreadable**
file counts as engaged (mirrors `bin/autoland.py`'s `AUTOLAND_KILL_FILE`
handling) — never treated as "absent, proceed". Every payload is a
`git archive` of **tracked** files only, so an untracked `.env` can never be
bundled by accident, scanned with `detect-secrets` before send; any hit
hard-stops. Provider terms are re-read weekly; a change to a provider's
data-use terms quarantines it (`status: quarantined`) until JP re-approves.
The weekly re-read is automated by `bin/terms-recheck.sh` (repo root), which
hash-compares each row's `source_url` body against `terms_hash` (`sha256:<hex>`) and surfaces
drift in the Monday jp-digest; quarantine itself stays JP's decision.

## Account and key handling

Per D6, JP alone creates provider accounts, accepts terms, and handles keys
(via `sops-edit`); no agent signs up or handles a key. Keys must never enter
the sealed local agent's environment — the local-agent lane's
`sanitized_env(forbid_keys=True)` already aborts a run if one is merely
present, and this lane must never weaken that boundary.
