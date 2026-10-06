"""Read the shared native/Python literal facts without code evaluation.

Only the three X-macro forms in core/runtime_facts.def are accepted. Context,
sampling and budget scopes remain explicit even when two values happen to be
equal. Launch profiles and Python-only UI defaults belong to runtime_defaults.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys
from types import MappingProxyType
from typing import Mapping


SOURCE = Path(__file__).resolve().parents[1] / "core" / "runtime_facts.def"
Value = int | float | str
_KEY = r"[A-Z][A-Z0-9_]*"
_INTEGER_LITERAL = r"(?:0|[1-9][0-9]*)"
_REAL_LITERAL = (r"[+-]?(?:0|[1-9][0-9]*)"
                 r"(?:\.[0-9]+(?:[eE][+-]?[0-9]+)?|[eE][+-]?[0-9]+)")
_TEXT_LITERAL = r'"(?:[\x20-\x21\x23-\x5b\x5d-\x7e]|\\["\\])*"'
_INTEGER = re.compile(
    rf"CACHEDMOE_RUNTIME_FACT_INTEGER\(\s*(uint32_t|uint64_t)\s*,\s*({_KEY})\s*,\s*({_INTEGER_LITERAL})\s*\)")
_REAL = re.compile(
    rf"CACHEDMOE_RUNTIME_FACT_REAL\(\s*({_KEY})\s*,\s*({_REAL_LITERAL})\s*\)")
_TEXT = re.compile(
    rf"CACHEDMOE_RUNTIME_FACT_TEXT\(\s*({_KEY})\s*,\s*({_TEXT_LITERAL})\s*\)")
_RESERVED = frozenset(("SOURCE", "FACTS", "TYPES"))


def _parse(text: str) -> tuple[dict[str, Value], dict[str, str]]:
    values: dict[str, Value] = {}
    types: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        integer = _INTEGER.fullmatch(line)
        real = _REAL.fullmatch(line)
        string = _TEXT.fullmatch(line)
        if integer:
            kind, key, literal = integer.groups()
            value: Value = int(literal, 10)
            bits = 32 if kind == "uint32_t" else 64
            if value >= 1 << bits:
                raise ValueError(f"line {number}: {key} exceeds {kind}")
        elif real:
            key, literal = real.groups()
            kind = "double"
            value = float(literal)
            if not math.isfinite(value):
                raise ValueError(f"line {number}: {key} must be finite")
        elif string:
            key, literal = string.groups()
            kind = "string_view"
            value = json.loads(literal)
        else:
            raise ValueError(f"line {number}: invalid runtime fact literal")
        if key in _RESERVED:
            raise ValueError(f"line {number}: reserved runtime fact key {key}")
        if key in values:
            raise ValueError(f"line {number}: duplicate runtime fact key {key}")
        values[key] = value
        types[key] = kind
    if not values:
        raise ValueError("runtime facts must contain at least one literal")
    return values, types


def parse_facts(text: str) -> Mapping[str, Value]:
    """Validate supplied literal text and return an immutable value mapping."""
    return MappingProxyType(_parse(text)[0])


def load_facts(path: Path = SOURCE) -> Mapping[str, Value]:
    """Read an authority file; environment variables do not redirect it."""
    return parse_facts(Path(path).read_text(encoding="utf-8"))


_values, _types = _parse(SOURCE.read_text(encoding="utf-8"))
FACTS: Mapping[str, Value] = MappingProxyType(_values)
TYPES: Mapping[str, str] = MappingProxyType(_types)
globals().update(FACTS)

# Both imports share the same facts and module identity.
sys.modules.setdefault("runtime_facts", sys.modules[__name__])
sys.modules.setdefault("tools.runtime_facts", sys.modules[__name__])
