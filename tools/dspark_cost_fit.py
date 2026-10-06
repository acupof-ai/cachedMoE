#!/usr/bin/env python3
"""Price confidence-prefix policies offline; never launch an engine.

This is an observational sensitivity model, not a replay or a measured speedup.
Confidence is available after draft execution, so every policy pays draft cost.
Only prefixes of recorded proposals can be evaluated. Alternative histories and
later routes are unknown after a policy changes an emitted token.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import runtime_defaults
import argparse
from collections import Counter
import json
import math
import statistics


def fit_verify(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['k'] + 1, []).append(row['verify_ms'])
    points = [(m, statistics.mean(cost)) for m, cost in sorted(groups.items())]
    if len(points) < 2:
        raise ValueError('need at least two measured verify row counts, including M=1')
    if 1 not in groups:
        raise ValueError('need a measured M=1 cycle; do not assume k=0 is free')
    xm = statistics.mean(x for x, _ in points)
    ym = statistics.mean(y for _, y in points)
    slope = sum((x-xm)*(y-ym) for x, y in points) / sum((x-xm)**2 for x, _ in points)
    intercept = ym - slope*xm
    if slope < 0 or intercept + slope <= 0:
        raise ValueError('observed verify costs do not support a positive row-cost model')
    return dict(intercept_ms=intercept, per_extra_row_ms=slope,
                measured=[dict(m=m, samples=len(groups[m]), mean_ms=y) for m, y in points],
                interpolated_m=[m for m in range(1, max(groups)+1) if m not in groups])


def prefix(confidence, threshold):
    k = 0
    for value in confidence:
        if value < threshold:
            break
        k += 1
    return k


def emitted(ranks, k, top_k):
    accepted = 0
    for rank in ranks[:k]:
        if rank >= top_k:
            break
        accepted += 1
    return accepted + 1


def analyze(rows, top_k, plain_ms_per_token):
    model = fit_verify(rows)
    active = [r for r in rows if r['k'] > 0]
    if not active:
        raise ValueError('no draft-bearing cycles')
    # Subtract only separately recorded first-seed work. Do not subtract the
    # draft itself when choosing k=0 after its confidence becomes available.
    draft = statistics.mean(r['draft_ms']-r.get('draft_seed_ms', 0) for r in active)
    commit = statistics.mean(r['commit_ms']+r['cpu_ms'] for r in active)
    thresholds = [-math.inf, -2, -1, 0, .5, 1, 2, 3, 4, 6, math.inf]
    policies = []
    for threshold in thresholds:
        ks = [prefix(r['confidence'], threshold) for r in active]
        outputs = sum(emitted(r['target_rank'], k, top_k) for r, k in zip(active, ks))
        verify = sum(model['intercept_ms']+model['per_extra_row_ms']*(k+1) for k in ks)
        cost = (len(active)*(draft+commit)+verify)/outputs
        policies.append(dict(threshold=str(threshold), cycles=len(active), selected_k=dict(Counter(ks)),
                             modeled_outputs=outputs, modeled_ms_per_token=cost,
                             modeled_gain_vs_plain=plain_ms_per_token/cost-1))
    fixed = policies[0]
    best = min(policies, key=lambda p: p['modeled_ms_per_token'])
    return dict(verify_model=model, paid_draft_ms=draft, commit_cpu_ms=commit,
                plain_ms_per_token=plain_ms_per_token, policies=policies,
                best_observational_policy=best,
                modeled_gain_over_fixed= fixed['modeled_ms_per_token']/best['modeled_ms_per_token']-1,
                limitations=[
                    'M=2 may be interpolated; M above the recorded draft length is not modeled.',
                    'A single end-of-generation M=1 sample is not a calibrated row-cost curve.',
                    'Policies pay full draft cost even if selected k=0; confidence is post-draft.',
                    'After the first rejection, ranks refer to the original draft prefix only.',
                    'Changing output changes KV, cache, acceptance, and future routes; this is not a replay.',
                    'Fit is in-sample and not validated on held-out prompts; do not enable a default from it.'])


def self_test():
    assert prefix([-1, 4], 0) == 0  # A good later row cannot rescue a bad prefix.
    assert prefix([2, .5, 4], 1) == 1
    assert emitted([0, 5, 0], 3, 4) == 2  # Never accept after a rejection.
    rows = [dict(k=0, verify_ms=100),
            dict(k=2, verify_ms=160, draft_ms=25, draft_seed_ms=0,
                 commit_ms=1, cpu_ms=0, confidence=[2, 3], target_rank=[0, 0])]
    report = analyze(rows, 4, 80)
    assert report['verify_model']['per_extra_row_ms'] == 30
    assert report['policies'][-1]['modeled_ms_per_token'] == 126
    assert report['policies'][0]['modeled_ms_per_token'] == 62
    print('dspark_cost_fit self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cycles', nargs='?')
    parser.add_argument('--plain-ms-per-token', type=float)
    parser.add_argument('--accept-top-k', type=int, default=runtime_defaults.ACCEPT_TOP_K)
    parser.add_argument('--out')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.cycles or not args.plain_ms_per_token or args.plain_ms_per_token <= 0:
        parser.error('provide cycle diagnostics and a positive measured plain ms/token')
    records = [json.loads(line) for line in Path(args.cycles).read_text().splitlines() if line.strip()]
    rows = [r for r in records if 'k' in r]
    report = analyze(rows, args.accept_top_k, args.plain_ms_per_token)
    text = json.dumps(report, indent=2) + '\n'
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
