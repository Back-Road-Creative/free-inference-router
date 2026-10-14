# Phase 1 compose — hardened Bifrost behind an allowlisting egress proxy

Offline runtime scaffold. No provider is admitted (`router/providers.json`
has no `status: admitted` entries yet) and no key exists anywhere in this
directory — `bifrost.config.json` references keys only as `env.<VAR>`
names, never values.

## Why network-layer egress control

Bifrost has no built-in egress/allowed-host feature (verified against its
client-config docs) — it will call whatever base URL a provider config
gives it. So the boundary can't live in Bifrost's own config; it has to
live in the network. This compose file implements the belt-and-suspenders
table in the Phase-0a plan (`docs/POLICY.md` §"Kill switch, outbound
payload"; plan `.data/plans/2026-08-29-local-first-free-inference-routing.md`
§6):

- `bifrost` sits on the `internal` Docker network only, which is declared
  `internal: true` — Docker gives that network no default route to the
  host's external interface, so a container on it alone cannot reach the
  internet at all, allowlist or not.
- `proxy` (tinyproxy) is the only service on **both** `internal` and
  `egress`. `bifrost`'s `HTTPS_PROXY`/`HTTP_PROXY` point at `proxy:8888`,
  so every outbound call is forced through it.
- `tinyproxy.conf` carries two *different* controls, not one — do not
  read `Allow` as the egress allowlist. `Allow 10.77.0.0/24` is tinyproxy's
  **client ACL**: it takes an IP/CIDR and decides which callers may use
  the proxy at all (the `internal` network's fixed subnet, set via
  `ipam` in the compose file, so the ACL can be exact). The actual
  **destination allowlist** is `Filter compose/tinyproxy.filter` plus
  `FilterDefaultDeny Yes` — one anchored regex per admitted host
  (`api.groq.com`, `generativelanguage.googleapis.com`, `openrouter.ai`).
  `ConnectPort 443` additionally limits CONNECT tunnels to HTTPS. A
  request to any other host is refused by the filter before it leaves
  the `egress` network.

Two independent layers have to both be misconfigured before a call
reaches an unadmitted host: the `internal` flag on Bifrost's network, and
tinyproxy's destination filter. Neither depends on Bifrost behaving
correctly.

Both containers are additionally hardened per the plan: pinned images
(tag **and** digest — this PR ships placeholder all-zero `sha256:...`
digests, Phase 2 pins the real ones), loopback-only published ports,
non-root users, read-only rootfs with explicit tmpfs, `cap_drop: [ALL]`,
`no-new-privileges`, and CPU/memory/pids bounds. No Docker socket mount,
no `privileged`.

## Phase-2 runtime fixes (2026-08-30, all verified by booting the stack offline)

The Phase-1 draft could not actually boot; every fix below was found by
running `docker compose up` with no keys and no admitted providers:

- **Images**: `maximhq/bifrost:1.2.3` and `alpine/tinyproxy:1.11.1` do not
  exist on Docker Hub. Now pinned: `maximhq/bifrost:v1.6.11` and
  `kalaksi/tinyproxy:1.7` (real digests, pulled by digest to verify).
- **No `command:` on bifrost**: the image's entrypoint busy-loops forever on
  any unrecognised argument, and `APP_DIR=/app/data` is already baked in.
- **`/app/data` tmpfs is `mode=1777`**: bifrost (uid 10001) must create its
  db files there before startup; the tmpfs mounts root-owned otherwise.
- **`providers` is a map, not an array** in `bifrost.config.json` (array
  fails unmarshal at bootstrap). `config_store.enabled: false` runs
  file-only mode, so the read-only config mount is safe.
- **Pricing + model parameters load from local files**
  (`framework.pricing.pricing_url` / `model_parameters_url`, `file://`
  scheme — it short-circuits URL validation and DNS): the default
  getbifrost.ai fetch is fatal at startup, and the sealed network
  correctly blocks it. `pricing.example.json` / `model-parameters.example.json`
  carry the admitted models at cost 0; zero-spend enforcement stays in
  `router/quota.py`, not here.
- **tinyproxy 1.11 syntax**: the `Filter` path must be double-quoted
  (unquoted is a fatal config syntax error) and `FilterType ere` replaces
  the deprecated `FilterExtended`.
- **`frontside` network**: docker cannot publish ports for a container
  only on an `internal` network, so the gateway sat healthy but
  unreachable. `frontside` is a bridge with IP masquerade disabled — the
  host reaches 127.0.0.1:8080, while outbound packets from that network
  are unroutable. Verified: `/v1/models` answers 200 from the host, and a
  direct in-container fetch of an external host (proxy env cleared) times
  out.

## proxy_config is mandatory on every provider

Bifrost is fasthttp-based and ignores `HTTPS_PROXY`/`HTTP_PROXY` env vars
(verified live: its own list-models calls timed out until the provider's
`proxy_config` pointed at `http://proxy:8888`). The env vars stay for any
stdlib-net path, but every provider entry MUST also carry
`"proxy_config": {"type": "http", "url": "http://proxy:8888"}` — a static
test enforces it.

## Running WITH keys (JP only) and the Groq admission smoke

`up-with-keys.sh` self-sudos, reads `GROQ_API_KEY` from the encrypted
store via `sops-get`, exports it only into the `docker compose up`
environment, and unsets it — the key is never echoed, never written to
disk, never left in the calling shell. `up-with-keys.sh down` tears the
stack down.

`smoke_groq.py` runs against the local gateway (`127.0.0.1:8080`) and
needs no client-side key. Synthetic text only. It checks the Phase-2
admission list — exact model id listed, chat, streaming, tool-call
argument shape, structured output, returned-model identity, usage
accounting — one PASS/FAIL line per check, exit 0 only when all pass.
Registry admission (`router/providers.json` status flip) happens in a
separate PR that quotes this smoke's output.

## Running the live canary (opt-in, NOT run in CI)

`compose/live-canary.sh` starts this stack and proves the boundary works
end to end: a request routed through `bifrost`/`proxy` to a non-admitted
host (`example.com`) must fail, and a request to an admitted host must
reach the network (a real 4xx from the provider counts as "reachable" —
this canary tests egress, not auth). It refuses to run unless
`ROUTER_LIVE_CANARY=1` is set, and it is never invoked by the pytest
suite (the router's `tests/test_egress_compose.py` asserts that). Run it
manually from the service directory:

```
ROUTER_LIVE_CANARY=1 bash compose/live-canary.sh
```

It requires a working `docker compose` and brings the stack up and back
down itself.

## Rollback

```
docker compose -f compose/docker-compose.yml down
```

There is no virtual key to disable yet in Phase 1 (no provider is
admitted); once Phase 2 adds one, disable it at the provider dashboard —
this compose file has no state of its own to roll back beyond `down`.
