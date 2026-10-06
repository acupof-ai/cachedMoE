"""Resolve cachedMoE controls without changing each consumer's value parser.

The canonical name wins by presence, including an empty value. Legacy names
are accepted only when the canonical name is absent. Scripts import this as
``runtime_env``; the package spelling points to the same warning state.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import sys
import threading
from collections.abc import Mapping, MutableMapping

CANONICAL_PREFIX = "CACHEDMOE_"
LEGACY_PREFIX = "DEEPMOE_"
PREFIXES = (CANONICAL_PREFIX, LEGACY_PREFIX)

sys.modules.setdefault("runtime_env", sys.modules[__name__])
sys.modules.setdefault("tools.runtime_env", sys.modules[__name__])

_warning_lock = threading.Lock()
_warned: set[tuple[str, str]] = set()


@dataclass(frozen=True)
class Resolution:
    value: str | None
    present: bool
    source: str
    selected_key: str | None


def is_control(key: str) -> bool:
    return key.startswith(PREFIXES)


def aliases(name: str) -> tuple[str, str]:
    for prefix in PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if not name:
        raise ValueError("environment control needs a nonempty suffix")
    return CANONICAL_PREFIX + name, LEGACY_PREFIX + name


def _warn_once(canonical: str, legacy: str, reason: str) -> None:
    with _warning_lock:
        marker = canonical, reason
        if marker in _warned:
            return
        _warned.add(marker)
        # Values may contain paths or secrets. Only names belong in a warning.
        if reason == "legacy":
            message = f"{legacy} is deprecated; use {canonical}"
        else:
            message = f"{canonical} and {legacy} conflict; {canonical} wins"
        print("cachedMoE environment: " + message, file=sys.stderr)


def resolve(name: str, environ: Mapping[str, str] | None = None,
            *, warn: bool = True) -> Resolution:
    env = os.environ if environ is None else environ
    canonical, legacy = aliases(name)
    if canonical in env:
        if warn and legacy in env and env[canonical] != env[legacy]:
            _warn_once(canonical, legacy, "conflict")
        return Resolution(env[canonical], True, "canonical", canonical)
    if legacy in env:
        if warn:
            _warn_once(canonical, legacy, "legacy")
        return Resolution(env[legacy], True, "legacy", legacy)
    return Resolution(None, False, "absent", None)


def getenv(name: str, default=None, environ: Mapping[str, str] | None = None):
    selected = resolve(name, environ)
    return selected.value if selected.present else default


def clear(environ: MutableMapping[str, str], *names: str) -> None:
    for name in names:
        for key in aliases(name):
            environ.pop(key, None)


def set_value(environ: MutableMapping[str, str], name: str, value: str) -> None:
    clear(environ, name)
    environ[aliases(name)[0]] = value


def setdefault(environ: MutableMapping[str, str], name: str, value: str):
    selected = resolve(name, environ)
    if selected.present:
        return selected.value
    environ[aliases(name)[0]] = value
    return value


def apply_overrides(environ: MutableMapping[str, str],
                    overrides: Mapping[str, str]) -> None:
    """Apply one explicit layer; aliases in that layer still resolve together.

    Removing both inherited names lets an explicit legacy CLI argument replace
    a generated canonical default. When the layer itself supplies both names,
    retain both so the normal canonical-presence rule decides the result.
    """
    for key in overrides:
        if is_control(key):
            clear(environ, key)
    environ.update(overrides)


def raw_controls(environ: Mapping[str, str]) -> dict[str, str]:
    return {k: v for k, v in environ.items() if is_control(k)}


def effective_controls(environ: Mapping[str, str], *, warn: bool = False
                       ) -> dict[str, Resolution]:
    # Preserve first-suffix order, so old-only perf-ledger rows render unchanged.
    names = dict.fromkeys(aliases(k)[0] for k in environ if is_control(k))
    return {k: resolve(k, environ, warn=warn) for k in names}


def canonicalized(environ: Mapping[str, str]) -> dict[str, str]:
    """Clear both families, then restore the chosen inherited controls.

    Non-project environment variables keep their original behavior. Intentional
    inherited controls (for example weighted-mask settings in a benchmark) are
    preserved; later explicit control layers can replace them.
    """
    selected = effective_controls(environ, warn=True)
    clean = {k: v for k, v in environ.items() if not is_control(k)}
    clean.update((k, r.value) for k, r in selected.items())
    return clean
