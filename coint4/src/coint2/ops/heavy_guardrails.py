"""Common heavy-run safety guardrails for shell and Python entrypoints."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from dataclasses import dataclass
from typing import Iterable

DEFAULT_ALLOW_ENV = "ALLOW_HEAVY_RUN"
DEFAULT_HOST_ALLOWLIST = ("85.198.90.128", "coint")
DEFAULT_MIN_RAM_GB = 28.0
DEFAULT_MIN_CPU = 8


@dataclass(frozen=True)
class HeavyGuardrailConfig:
    """Declarative heavy-run constraints."""

    entrypoint: str
    allow_env: str = DEFAULT_ALLOW_ENV
    host_allowlist: tuple[str, ...] = DEFAULT_HOST_ALLOWLIST
    min_ram_gb: float = DEFAULT_MIN_RAM_GB
    min_cpu: int = DEFAULT_MIN_CPU


@dataclass(frozen=True)
class HeavyGuardrailResult:
    """Guardrail evaluation result."""

    config: HeavyGuardrailConfig
    allow_value: str
    host_candidates: tuple[str, ...]
    cpu_count: int
    ram_gb: float | None
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def parse_host_allowlist(raw: str | None, *, default: Iterable[str] = DEFAULT_HOST_ALLOWLIST) -> tuple[str, ...]:
    """Parse comma-separated hostname/IP allowlist."""
    value = str(raw or "").strip()
    if not value:
        return tuple(str(item).strip().lower() for item in default if str(item).strip())
    parsed = [part.strip().lower() for part in value.split(",") if part.strip()]
    return tuple(parsed)


def detect_total_ram_gb() -> float | None:
    """Best-effort RAM detection in GiB."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    mem_kib = float(line.split()[1])
                    return mem_kib / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and phys_pages > 0:
            return (float(page_size) * float(phys_pages)) / (1024.0**3)
    except Exception:  # noqa: BLE001
        pass
    return None


def detect_host_candidates() -> tuple[str, ...]:
    """Collect host identifiers (hostname/fqdn/local IPs) for allowlist matching."""
    candidates: set[str] = set()

    try:
        hostname = socket.gethostname().strip().lower()
        if hostname:
            candidates.add(hostname)
    except Exception:  # noqa: BLE001
        pass

    try:
        fqdn = socket.getfqdn().strip().lower()
        if fqdn:
            candidates.add(fqdn)
    except Exception:  # noqa: BLE001
        pass

    for host in tuple(candidates):
        try:
            _, _, addrs = socket.gethostbyname_ex(host)
            for addr in addrs:
                addr = str(addr).strip().lower()
                if addr:
                    candidates.add(addr)
        except Exception:  # noqa: BLE001
            continue

    try:
        for item in socket.getaddrinfo(socket.gethostname(), None):
            addr = str(item[4][0]).strip().lower()
            if addr:
                candidates.add(addr)
    except Exception:  # noqa: BLE001
        pass

    return tuple(sorted(candidates))


def evaluate_heavy_guardrails(config: HeavyGuardrailConfig, *, environ: dict[str, str] | None = None) -> HeavyGuardrailResult:
    """Evaluate heavy-run guardrails and return a structured result."""
    env = os.environ if environ is None else environ
    failures: list[str] = []

    allow_value = str(env.get(config.allow_env, "")).strip()
    if allow_value != "1":
        failures.append(
            f"{config.allow_env} must be exactly '1' for heavy execution (current: {allow_value or '<unset>'})."
        )

    host_candidates = detect_host_candidates()
    allowlist = {item.strip().lower() for item in config.host_allowlist if item.strip()}
    if allowlist and allowlist.isdisjoint(host_candidates):
        failures.append(
            "hostname/IP is outside allowlist "
            f"(allowlist={sorted(allowlist)}, detected={list(host_candidates) or ['<unknown>']})."
        )

    cpu_count = int(os.cpu_count() or 0)
    if cpu_count < int(config.min_cpu):
        failures.append(f"CPU cores below minimum ({cpu_count} < {int(config.min_cpu)}).")

    ram_gb = detect_total_ram_gb()
    if ram_gb is None:
        failures.append("unable to detect total RAM (fail-closed).")
    elif ram_gb < float(config.min_ram_gb):
        failures.append(f"RAM below minimum ({ram_gb:.1f} GiB < {float(config.min_ram_gb):.1f} GiB).")

    return HeavyGuardrailResult(
        config=config,
        allow_value=allow_value,
        host_candidates=host_candidates,
        cpu_count=cpu_count,
        ram_gb=ram_gb,
        failures=tuple(failures),
    )


def format_guardrail_failure(result: HeavyGuardrailResult) -> str:
    """Build actionable error text for operators."""
    reasons = "\n".join(f"- {line}" for line in result.failures)
    allowlist = ",".join(result.config.host_allowlist)
    return (
        f"[heavy-guardrails] BLOCKED: {result.config.entrypoint}\n"
        f"{reasons}\n"
        "Fix:\n"
        f"- export {result.config.allow_env}=1\n"
        f"- run on allowlisted host ({allowlist}) or override HEAVY_HOSTNAME_ALLOWLIST explicitly\n"
        f"- ensure host resources >= {int(result.config.min_cpu)} CPU and >= {float(result.config.min_ram_gb):.1f} GiB RAM"
    )


def ensure_heavy_run_allowed(config: HeavyGuardrailConfig, *, environ: dict[str, str] | None = None) -> HeavyGuardrailResult:
    """Raise SystemExit on guardrail violation."""
    result = evaluate_heavy_guardrails(config, environ=environ)
    if not result.passed:
        raise SystemExit(format_guardrail_failure(result))
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate heavy-run guardrails.")
    parser.add_argument("--entrypoint", required=True, help="Entrypoint identifier for logs.")
    parser.add_argument("--allow-env", default=DEFAULT_ALLOW_ENV, help="Environment variable that must equal '1'.")
    parser.add_argument(
        "--allowlist",
        default=os.environ.get("HEAVY_HOSTNAME_ALLOWLIST", ",".join(DEFAULT_HOST_ALLOWLIST)),
        help="Comma-separated hostname/IP allowlist.",
    )
    parser.add_argument(
        "--min-ram-gb",
        type=float,
        default=float(os.environ.get("HEAVY_MIN_RAM_GB", DEFAULT_MIN_RAM_GB)),
        help="Minimum required RAM in GiB.",
    )
    parser.add_argument(
        "--min-cpu",
        type=int,
        default=int(os.environ.get("HEAVY_MIN_CPU", DEFAULT_MIN_CPU)),
        help="Minimum required CPU core count.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    config = HeavyGuardrailConfig(
        entrypoint=str(args.entrypoint),
        allow_env=str(args.allow_env),
        host_allowlist=parse_host_allowlist(str(args.allowlist)),
        min_ram_gb=float(args.min_ram_gb),
        min_cpu=int(args.min_cpu),
    )
    result = evaluate_heavy_guardrails(config)

    payload = {
        "entrypoint": config.entrypoint,
        "passed": result.passed,
        "allow_env": config.allow_env,
        "allow_value": result.allow_value,
        "host_allowlist": list(config.host_allowlist),
        "host_candidates": list(result.host_candidates),
        "cpu_count": result.cpu_count,
        "min_cpu": int(config.min_cpu),
        "ram_gb": result.ram_gb,
        "min_ram_gb": float(config.min_ram_gb),
        "failures": list(result.failures),
    }

    if args.json:
        stream = sys.stdout if result.passed else sys.stderr
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True), file=stream)
    elif result.passed:
        print(
            "[heavy-guardrails] OK: "
            f"{config.entrypoint} host={','.join(result.host_candidates) or '<unknown>'} "
            f"cpu={result.cpu_count} ram_gb={(result.ram_gb if result.ram_gb is not None else 'unknown')}"
        )
    else:
        print(format_guardrail_failure(result), file=sys.stderr)

    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
