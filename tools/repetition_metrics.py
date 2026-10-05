#!/usr/bin/env python3
"""Token-level repetition gates for JSON-lines generation events.

Each done/error boundary starts a new output. Prompts and tokenisation replies
are excluded. A short-period loop needs at least four complete repeats and 16
tokens; we search suffixes in each rolling 128-token window. This avoids treating
two ordinary repeated phrases as a loop. Raw token runs have a separate <=3 gate.
These structural metrics do not measure semantic correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def metrics(tokens: list[int], *, window: int = 128, max_period: int = 8) -> dict:
    longest = run = 0
    previous = None
    for token in tokens:
        run = run + 1 if token == previous else 1
        longest = max(longest, run)
        previous = token
    short = None
    for end in range(16, len(tokens) + 1):
        for period in range(1, max_period + 1):
            length = max(16, 4 * period)
            if length > min(window, end):
                continue
            start = end - length
            if all(tokens[i] == tokens[i - period] for i in range(start + period, end)):
                short = dict(start=start, end=end, period=period, tokens=length)
                break
        if short is not None:
            break
    counts4 = Counter(tuple(tokens[i:i + 4]) for i in range(max(0, len(tokens) - 3)))
    n4 = max(0, len(tokens) - 3)
    n2 = max(0, len(tokens) - 1)
    return dict(tokens=len(tokens), longest_same_token_run=longest,
                short_period_loop=short is not None, first_short_period=short,
                repeated_4gram_fraction=sum(n - 1 for n in counts4.values()) / n4 if n4 else 0.0,
                distinct_2=len({tuple(tokens[i:i + 2]) for i in range(n2)}) / n2 if n2 else 0.0,
                no_loop=longest <= 3 and short is None)


def from_events(path: str | Path) -> dict:
    outputs, ids = [], []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid event JSON") from exc
            if event.get("event") == "token":
                token = event.get("id")
                if not isinstance(token, int) or isinstance(token, bool) or token < 0:
                    raise ValueError(f"{path}:{number}: invalid token id")
                ids.append(token)
            elif event.get("event") in ("done", "error") and ids:
                outputs.append(metrics(ids))
                ids = []
    if ids:
        outputs.append(metrics(ids))
    return dict(source=str(path), outputs=outputs, tokens=sum(r["tokens"] for r in outputs),
                no_loop=bool(outputs) and all(r["no_loop"] for r in outputs))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--check", action="store_true", help="exit 1 if any output fails")
    args = parser.parse_args()
    report = dict(definition=dict(window=128, max_period=8, min_cycles=4, min_loop_tokens=16,
                                  max_same_token_run=3), runs=[from_events(p) for p in args.events])
    encoded = json.dumps(report, indent=2) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded)
    else:
        print(encoded, end="")
    return int(args.check and any(not r["no_loop"] for r in report["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
