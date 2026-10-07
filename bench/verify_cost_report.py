#!/usr/bin/env python3
"""Attribute recorded verify costs without adding nested host timers together."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from spec_diagnostics import verify_report


def summarize(diagnostics, events):
    done = [row for row in events if row.get('event') == 'done']
    count = sum(row['speculation']['cycles'] for row in done)
    if not count or diagnostics[0].get('schema') != 1:
        raise ValueError('require complete native speculative events and a diagnostic header')
    rows = diagnostics[1:1 + count]
    if len(rows) != count or any('k' not in row for row in rows):
        raise ValueError('missing native cycles or multiple headers')
    offset = 0
    for turn in done:
        n = turn['speculation']['cycles']
        segment = rows[offset:offset+n]
        if sum(row['emitted'] for row in segment) != turn['decode_steps']:
            raise ValueError('emitted count does not match the turn')
        for previous, current in zip(segment, segment[1:]):
            if current['position'] != previous['position'] + previous['emitted']:
                raise ValueError('missing or reordered cycles')
        offset += n
    names = sorted(set().union(*(r['verify_host_ms'] for r in rows)))
    host = {name: sum(r['verify_host_ms'].get(name, 0) for r in rows)/count
            for name in names}
    cost = {name: sum(row['speculation'][name] for row in done)/count
            for name in ('draft_ms', 'verify_ms', 'commit_ms', 'cpu_ms')}
    total = sum(cost.values())
    wait_ms = sum(row.get('expert_io_wait', {}).get('ms', 0) for row in rows)
    failures = {name: sum(row.get('load_failures', {}).get(name, 0) for row in rows)
                for name in ('p0_reserve', 'p0_submit', 'p0_io', 'fill')}
    return dict(cycles=count, turns=len(done), mean_cycle_ms=cost,
                accounted_cycle_ms=total, verify_fraction=cost['verify_ms']/total,
                expert_io_wait_ms=wait_ms, load_failures=failures,
                mean_verify_host_ms=host,
                host_timers_are_additive=False,
                nested_timers=['record_layers_tail includes layer fences and Engram waits',
                               'fence_wait accumulates waits inside and outside record_layers_tail',
                               'engram_issue_land overlaps record_layers_tail'],
                production_speed_winner=False)


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(load(args.diagnostics), load(args.events))
    if args.trace:
        result['dispatch_trace'] = verify_report(args.trace)
    result['sources'] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (args.diagnostics, args.events, args.trace) if path}
    with args.out.open('x') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
