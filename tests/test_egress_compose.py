"""Static canaries for the Phase-1 egress compose stack.

Bifrost (the gateway) has no egress/allowed-host feature of its own
(verified against its client-config docs), so egress restriction here is a
network-layer control: Bifrost sits on an ``internal: true`` Docker network
with no default route, and the only way out is a tiny allowlisting
forward-proxy (tinyproxy) that straddles both networks. These tests assert
the compose file's structure and the two sidecar configs match the hardening
table in ``docs/POLICY.md``. They do NOT touch the network; the live canary
(``compose/live-canary.sh``) is opt-in and documented in ``compose/README.md``.

Uses ``yaml`` (stdlib does not parse YAML): a real parser is preferred over
hand-rolled line matching for compose's nested structure, so PyYAML is a
dev-only dependency; the router itself stays dependency-free.
"""

from __future__ import annotations

import ipaddress
import json
import re
from pathlib import Path
from typing import Any

import yaml

ROUTER_DIR = Path(__file__).resolve().parents[1]
COMPOSE_DIR = ROUTER_DIR / "compose"
COMPOSE_PATH = COMPOSE_DIR / "docker-compose.yml"
BIFROST_CONFIG_PATH = COMPOSE_DIR / "bifrost.config.json"
TINYPROXY_CONF_PATH = COMPOSE_DIR / "tinyproxy.conf"
TINYPROXY_FILTER_PATH = COMPOSE_DIR / "tinyproxy.filter"
README_PATH = COMPOSE_DIR / "README.md"
LIVE_CANARY_PATH = COMPOSE_DIR / "live-canary.sh"

ADMITTED_PROXY_HOSTS = {
    "api.groq.com",
    "generativelanguage.googleapis.com",
    "openrouter.ai",
}
ADMITTED_FILTER_PATTERNS = {f"^{re.escape(h)}$" for h in ADMITTED_PROXY_HOSTS}

KEY_SHAPE_PATTERNS = [
    re.compile(r"gsk_[A-Za-z0-9]"),
    re.compile(r"AIzaSy[A-Za-z0-9_-]"),
    re.compile(r"sk-or-[A-Za-z0-9]"),
    re.compile(r"nvapi-[A-Za-z0-9]"),
]

Network = ipaddress.IPv4Network | ipaddress.IPv6Network


def _compose() -> dict[str, Any]:
    data = yaml.safe_load(COMPOSE_PATH.read_text())
    assert isinstance(data, dict)
    return data


def _service(name: str) -> dict[str, Any]:
    svc = _compose()["services"][name]
    assert isinstance(svc, dict)
    return svc


def _bifrost() -> dict[str, Any]:
    return _service("bifrost")


def _proxy() -> dict[str, Any]:
    return _service("proxy")


def _bifrost_config() -> dict[str, Any]:
    cfg = json.loads(BIFROST_CONFIG_PATH.read_text())
    assert isinstance(cfg, dict)
    return cfg


def _internal_subnet() -> Network:
    """The fixed subnet declared for the `internal` network via `ipam`."""
    net = _compose()["networks"]["internal"]
    subnet = net["ipam"]["config"][0]["subnet"]
    return ipaddress.ip_network(subnet)


def _parse_allow_values(conf_text: str) -> list[str]:
    return re.findall(r"^Allow\s+(\S+)", conf_text, re.MULTILINE)


def _client_acl_is_valid(conf_text: str, expected_subnet: Network) -> bool:
    """tinyproxy's `Allow` is a CLIENT ACL (source IP/CIDR), not a
    destination allowlist. This validates that ONLY that use is present:
    exactly one `Allow` value, parseable as an IPv4/IPv6 network, and
    equal to the expected subnet. A hostname (or any non-network string)
    in `Allow` fails this, on purpose: that is the bug this helper
    exists to catch.
    """
    values = _parse_allow_values(conf_text)
    if len(values) != 1:
        return False
    try:
        net = ipaddress.ip_network(values[0], strict=True)
    except ValueError:
        return False
    return net == expected_subnet


# --- image pinning ---------------------------------------------------


def test_no_image_tag_is_latest() -> None:
    compose = _compose()
    for name, svc in compose["services"].items():
        image = svc.get("image", "")
        tag = image.split("@")[0]
        assert not tag.endswith(":latest"), f"{name} pins image to latest: {image!r}"


def test_every_image_has_a_sha256_digest() -> None:
    compose = _compose()
    for name, svc in compose["services"].items():
        image = svc.get("image", "")
        assert "@sha256:" in image, f"{name} image {image!r} has no @sha256 digest"


def test_no_image_digest_is_the_allzero_placeholder() -> None:
    """Phase 2 pinned real digests; the 64-zero Phase-1 placeholder must not return."""
    compose = _compose()
    for name, svc in compose["services"].items():
        image = svc.get("image", "")
        assert "@sha256:" + "0" * 64 not in image, f"{name} has the placeholder digest"


def test_every_image_digest_is_exactly_64_hex_chars() -> None:
    compose = _compose()
    for name, svc in compose["services"].items():
        image = svc.get("image", "")
        assert "@sha256:" in image, f"{name} image {image!r} has no @sha256 digest"
        digest_hex = image.split("@sha256:", 1)[1]
        assert re.fullmatch(r"[0-9a-f]+", digest_hex), (
            f"{name} digest has non-hex chars: {digest_hex!r}"
        )
        assert len(digest_hex) == 64, (
            f"{name} digest is {len(digest_hex)} hex chars, want 64: {digest_hex!r}"
        )


# --- network binding ---------------------------------------------------


def test_bifrost_bound_to_loopback_only() -> None:
    bifrost = _bifrost()
    ports = bifrost.get("ports", [])
    assert ports, "bifrost declares no ports"
    for p in ports:
        assert str(p).startswith("127.0.0.1:"), f"bifrost port {p!r} is not loopback"


def test_ui_bound_to_loopback_only() -> None:
    compose = _compose()
    # UI may be a distinct service or exposed off bifrost; either way every
    # published port across the whole file must be loopback-bound.
    for name, svc in compose["services"].items():
        for p in svc.get("ports", []):
            assert str(p).startswith("127.0.0.1:"), f"{name} port {p!r} is not loopback"


def test_bifrost_network_list_is_internal_plus_noegress_frontside() -> None:
    """frontside exists only for host->gateway loopback publishing (docker
    cannot publish ports on an internal-only network); masquerade is off so
    it carries no outbound path. The proxy on `internal` stays the only exit."""
    bifrost = _bifrost()
    assert set(bifrost.get("networks", [])) == {"internal", "frontside"}
    nets = _compose().get("networks", {})
    fs = nets.get("frontside", {})
    masquerade = fs.get("driver_opts", {}).get(
        "com.docker.network.bridge.enable_ip_masquerade"
    )
    assert masquerade == "false"
    assert not fs.get("internal"), "frontside must NOT be internal: publishing breaks"


def test_internal_network_has_no_default_route() -> None:
    compose = _compose()
    internal_net = compose["networks"]["internal"]
    assert internal_net.get("internal") is True


def test_proxy_is_on_both_networks() -> None:
    proxy = _proxy()
    assert set(proxy.get("networks", [])) == {"internal", "egress"}


def test_no_service_mounts_docker_socket() -> None:
    compose = _compose()
    for name, svc in compose["services"].items():
        for vol in svc.get("volumes", []):
            assert "docker.sock" not in str(vol), f"{name} mounts the docker socket"


def test_no_service_is_privileged() -> None:
    compose = _compose()
    for name, svc in compose["services"].items():
        assert not svc.get("privileged", False), f"{name} runs privileged"


# --- hardening flags (per-service) --------------------------------------


def test_bifrost_runs_non_root() -> None:
    bifrost = _bifrost()
    user = bifrost.get("user")
    assert user and user not in ("root", "0", "0:0"), f"bifrost user={user!r}"


def test_bifrost_rootfs_is_read_only() -> None:
    bifrost = _bifrost()
    assert bifrost.get("read_only") is True


def test_bifrost_declares_tmpfs_for_writes() -> None:
    bifrost = _bifrost()
    assert bifrost.get("tmpfs"), "bifrost has no tmpfs entries despite read_only rootfs"


def test_bifrost_drops_all_capabilities() -> None:
    bifrost = _bifrost()
    assert bifrost.get("cap_drop") == ["ALL"]


def test_bifrost_sets_no_new_privileges() -> None:
    bifrost = _bifrost()
    assert "no-new-privileges:true" in bifrost.get("security_opt", [])


def test_bifrost_has_cpu_memory_and_pids_bounds() -> None:
    bifrost = _bifrost()
    limits = bifrost.get("deploy", {}).get("resources", {}).get("limits", {})
    has_mem = bool(bifrost.get("mem_limit")) or bool(limits.get("memory"))
    has_cpu = bool(bifrost.get("cpus")) or bool(limits.get("cpus"))
    assert has_mem, "bifrost has no memory limit"
    assert has_cpu, "bifrost has no cpu limit"
    assert bifrost.get("pids_limit"), "bifrost has no pids_limit"


def test_proxy_also_hardened() -> None:
    proxy = _proxy()
    assert proxy.get("read_only") is True
    assert proxy.get("cap_drop") == ["ALL"]
    assert "no-new-privileges:true" in proxy.get("security_opt", [])
    user = proxy.get("user")
    assert user and user not in ("root", "0", "0:0"), f"proxy user={user!r}"
    assert proxy.get("pids_limit"), "proxy has no pids_limit"


def _environment(svc: dict[str, Any]) -> dict[str, str]:
    env = svc.get("environment", {})
    if isinstance(env, list):
        return dict(item.split("=", 1) for item in env if "=" in item)
    assert isinstance(env, dict)
    return env


def test_bifrost_reaches_upstreams_via_the_proxy() -> None:
    env = _environment(_bifrost())
    assert env.get("HTTPS_PROXY") == "http://proxy:8888", env.get("HTTPS_PROXY")


# --- tinyproxy egress control ---------------------------------------------
#
# tinyproxy's `Allow` directive is a CLIENT ACL (source IP/CIDR): it
# decides which callers may use the proxy, NOT which destinations the
# proxy will reach. Destination allowlisting is `Filter <file>` +
# `FilterDefaultDeny Yes`. A conf with `Allow api.groq.com` (a hostname
# where `Allow` expects a network) filters nothing: `FilterDefaultDeny`
# is inert without a `Filter` directive. These tests are written so they
# cannot pass on the wrong directive: they validate the CLIENT ACL is a
# real CIDR equal to the internal subnet, and separately that a real
# destination FILTER exists, is mounted read-only, denies by default, and
# contains exactly the three anchored admitted-host patterns.


def test_tinyproxy_client_acl_is_exactly_the_internal_subnet() -> None:
    text = TINYPROXY_CONF_PATH.read_text()
    assert _client_acl_is_valid(text, _internal_subnet())


def test_client_acl_validator_rejects_a_hostname_in_allow() -> None:
    # The exact bug this whole section guards against: `Allow api.groq.com`
    # looks plausible but is a no-op client ACL, not a destination filter.
    bad_conf = "Allow api.groq.com\nFilterDefaultDeny Yes\n"
    assert not _client_acl_is_valid(bad_conf, _internal_subnet())


def test_client_acl_validator_rejects_more_than_one_allow_line() -> None:
    bad_conf = "Allow 10.77.0.0/24\nAllow 10.78.0.0/24\n"
    assert not _client_acl_is_valid(bad_conf, _internal_subnet())


def test_tinyproxy_filter_directive_present_and_mounted_read_only() -> None:
    text = TINYPROXY_CONF_PATH.read_text()
    m = re.search(r"^Filter\s+(\S+)\s*$", text, re.MULTILINE)
    assert m, "tinyproxy.conf has no Filter directive; FilterDefaultDeny is inert"
    # tinyproxy 1.11 requires the path in double quotes; compare unquoted
    filter_path_in_container = m.group(1).strip('"')

    proxy_volumes = _proxy().get("volumes", [])
    mounted_ro = any(
        str(v).startswith("./tinyproxy.filter:")
        and str(v).endswith(":ro")
        and filter_path_in_container in str(v)
        for v in proxy_volumes
    )
    assert mounted_ro, (
        f"compose does not mount tinyproxy.filter read-only at "
        f"{filter_path_in_container}"
    )


def test_tinyproxy_filter_default_deny_yes() -> None:
    text = TINYPROXY_CONF_PATH.read_text()
    assert re.search(r"^FilterDefaultDeny\s+Yes\s*$", text, re.MULTILINE)


def test_tinyproxy_filter_file_is_exactly_the_three_anchored_patterns() -> None:
    lines = [
        line.strip()
        for line in TINYPROXY_FILTER_PATH.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert set(lines) == ADMITTED_FILTER_PATTERNS, lines
    for pattern in lines:
        assert pattern.startswith("^") and pattern.endswith("$"), (
            f"unanchored filter pattern: {pattern!r}"
        )


def test_tinyproxy_connect_port_is_443_only() -> None:
    text = TINYPROXY_CONF_PATH.read_text()
    assert re.search(r"^ConnectPort\s+443\s*$", text, re.MULTILINE)


def test_tinyproxy_deny_by_default_no_upstream() -> None:
    text = TINYPROXY_CONF_PATH.read_text()
    assert not re.search(r"^Upstream\b", text, re.MULTILINE), (
        "tinyproxy.conf declares an Upstream (should deny-by-default)"
    )


def test_tinyproxy_listens_on_all_interfaces_port_8888() -> None:
    text = TINYPROXY_CONF_PATH.read_text()
    assert re.search(r"^Listen\s+0\.0\.0\.0\s*$", text, re.MULTILINE)
    assert re.search(r"^Port\s+8888\s*$", text, re.MULTILINE)


def test_tinyproxy_disables_via_header() -> None:
    text = TINYPROXY_CONF_PATH.read_text()
    assert re.search(r"^DisableViaHeader\s+Yes\s*$", text, re.MULTILINE)


# --- bifrost.config.json flags -------------------------------------------


def test_bifrost_config_disables_content_logging() -> None:
    cfg = _bifrost_config()
    assert cfg["disable_content_logging"] is True


def test_bifrost_config_forbids_per_request_storage_override() -> None:
    cfg = _bifrost_config()
    assert cfg["allow_per_request_content_storage_override"] is False


def test_bifrost_config_does_not_retain_content_in_object_storage() -> None:
    cfg = _bifrost_config()
    assert cfg["retain_content_in_object_storage"] is False


def test_bifrost_config_log_retention_days_present_and_at_most_14() -> None:
    cfg = _bifrost_config()
    assert "log_retention_days" in cfg, "log_retention_days absent (default is 365)"
    assert cfg["log_retention_days"] <= 14


def test_bifrost_config_has_no_observability_connectors() -> None:
    cfg = _bifrost_config()
    for key in ("observability", "observability_connectors", "connectors"):
        if key in cfg:
            assert not cfg[key], f"{key} is non-empty: {cfg[key]!r}"


def test_bifrost_config_disables_semantic_caching() -> None:
    cfg = _bifrost_config()
    assert cfg.get("semantic_cache_enabled") is False


def test_bifrost_config_providers_are_env_keyed_and_pinned() -> None:
    """Each configured provider: exactly one key, an env.* reference (never a
    value), and a pinned model list; no wildcard anywhere."""
    cfg = _bifrost_config()
    providers = cfg.get("providers", {})
    assert set(providers) == {"groq", "openrouter", "gemini"}, sorted(providers)
    expected_models = {
        "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        "openrouter": [
            "inclusionai/ling-3.0-flash-fin:free",
            "poolside/laguna-s-2.1:free",
        ],
        "gemini": ["gemini-3.5-flash", "gemini-3.5-flash-lite"],
    }
    for name, provider in providers.items():
        keys = provider["keys"]
        assert len(keys) == 1, name
        assert keys[0]["value"] == f"env.{name.upper()}_API_KEY", name
        assert keys[0]["models"] == expected_models[name], name
        assert "*" not in keys[0]["models"], name


def test_every_bifrost_provider_pins_proxy_config_to_the_egress_proxy() -> None:
    """Bifrost is fasthttp-based and IGNORES HTTPS_PROXY env vars (verified
    live 2026-08-31: list-models timed out until proxy_config was set). The
    env vars stay as belt; this per-provider pin is the suspenders. Without
    it a provider's traffic silently tries (and on this network, fails) to
    bypass the filtering proxy."""
    cfg = _bifrost_config()
    for name, provider in cfg.get("providers", {}).items():
        pc = provider.get("proxy_config")
        assert pc, f"provider {name} has no proxy_config"
        assert pc == {"type": "http", "url": "http://proxy:8888"}, f"{name}: {pc}"


def test_compose_groq_key_passthrough_defaults_empty() -> None:
    """GROQ_API_KEY reaches the container only via ${GROQ_API_KEY:-}: an
    empty default so the offline boot needs no key, and never a literal."""
    env = _environment(_bifrost())
    assert env.get("GROQ_API_KEY") == "${GROQ_API_KEY:-}", env.get("GROQ_API_KEY")
    assert env.get("OPENROUTER_API_KEY") == "${OPENROUTER_API_KEY:-}"
    assert env.get("GEMINI_API_KEY") == "${GEMINI_API_KEY:-}"


def test_launcher_reads_key_from_store_and_never_echoes_it() -> None:
    text = (COMPOSE_DIR / "up-with-keys.sh").read_text()
    assert "GROQ_API_KEY OPENROUTER_API_KEY GEMINI_API_KEY" in text, (
        "launcher must load all three keys"
    )
    assert "sops-get" in text
    for line in text.splitlines():
        if line.strip().startswith("echo"):
            assert "$GROQ_API_KEY" not in line and "${GROQ_API_KEY" not in line


# --- secret hygiene -------------------------------------------------------


def test_no_key_value_appears_anywhere_under_compose() -> None:
    for pattern in KEY_SHAPE_PATTERNS:
        for f in COMPOSE_DIR.iterdir():
            if f.is_file():
                assert not pattern.search(f.read_text()), f"key shape in {f.name}"


def test_no_env_file_directive_in_compose() -> None:
    text = COMPOSE_PATH.read_text()
    assert "env_file" not in text, "compose must not reference an .env file"


def test_bifrost_config_keys_are_env_references_only() -> None:
    text = BIFROST_CONFIG_PATH.read_text()
    for pattern in KEY_SHAPE_PATTERNS:
        assert not pattern.search(text), "key-shaped value in bifrost.config.json"


# --- supporting files exist and are well-formed --------------------------


def test_readme_exists_and_documents_topology() -> None:
    assert README_PATH.exists()
    text = README_PATH.read_text().lower()
    for term in ("internal", "proxy", "canary", "rollback"):
        assert term in text, f"README missing discussion of {term!r}"


def test_live_canary_refuses_without_opt_in_env_var() -> None:
    assert LIVE_CANARY_PATH.exists()
    text = LIVE_CANARY_PATH.read_text()
    assert "ROUTER_LIVE_CANARY" in text


def test_live_canary_not_invoked_by_any_test_in_this_file() -> None:
    # Sanity guard: this test module must never shell out to live-canary.sh.
    # (Checked by grepping for the two stdlib mechanisms that could do it,
    # split so the check string doesn't match its own source line.)
    this_file = Path(__file__).read_text()
    forbidden = ["sub" + "process", "os." + "system"]
    for token in forbidden:
        assert token not in this_file, f"test module must not shell out ({token})"
