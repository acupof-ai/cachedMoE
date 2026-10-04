#!/usr/bin/env python3
"""Analyze opt-in main-path DSpark diagnostics, without launching a GPU job.

LRU is a fixed-route, synchronous-fill sensitivity model. It omits background
fills, execution guards, and the change in routing caused by a different mask.
It is not an engine replay or a speed/quality prediction.
"""
import argparse
from collections import OrderedDict, Counter
import json
import math
from pathlib import Path

EXPERT_BYTES = 18_800_640


def read_trace(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not records or records[0].get('schema') != 1:
        raise ValueError('expected a schema-1 diagnostic header')
    header, rows = records[0], records[1:]
    if not rows:
        raise ValueError('no speculative cycles')
    last = None
    for row in rows:
        if 'k' not in row:
            raise ValueError('multiple headers in appended trace; analyze one process segment')
        k = row['k']
        if not 0 <= k <= 5 or len(row['confidence']) != k or len(row['target_rank']) != k:
            raise ValueError('invalid confidence/rank geometry')
        if len(row['target_routes']) != 40 * (k + 1) * 6:
            raise ValueError('invalid main route geometry')
        if len(row['draft_routes']) != 45 or (k and not row['draft_routes_valid']):
            raise ValueError('draft routes unavailable; collect with DEEPMOE_DSPARK_MEGA=0')
        if any(not math.isfinite(x) for x in row['confidence']):
            raise ValueError('nonfinite confidence')
        if any(not 0 <= e < 384 for e in row['target_routes']) or any(not 0 <= e < 128 for e in row['draft_routes']):
            raise ValueError('invalid expert id')
        if last is not None and row['position'] != last:
            raise ValueError('trace spans a reset or has missing cycles; analyze one contiguous segment')
        last = row['position'] + row['emitted']
    if not any(row['draft_routes_valid'] for row in rows):
        raise ValueError('no recorded serial draft; cannot distinguish absent draft from mega diagnostics')
    return header, rows


def confidence_report(rows, accept_k):
    bins = [-math.inf, 0, 1, 2, 3, 4, 6, math.inf]
    by_bin = [Counter() for _ in range(len(bins) - 1)]
    positions = [Counter() for _ in range(5)]
    for row in rows:
        prefix = True
        for j, (c, rank) in enumerate(zip(row['confidence'], row['target_rank'])):
            hit = rank < accept_k
            b = next(i for i in range(len(by_bin)) if bins[i] <= c < bins[i + 1])
            for bucket in (by_bin[b], positions[j]):
                bucket['rows'] += 1
                bucket['topk'] += hit
                bucket['reachable'] += prefix
                bucket['reachable_topk'] += prefix and hit
                bucket['confidence_sum'] += c
            prefix = prefix and hit
    def record(bucket):
        return dict(bucket) | dict(
            acceptance=bucket['topk'] / bucket['rows'] if bucket['rows'] else None,
            conditional_acceptance=bucket['reachable_topk'] / bucket['reachable'] if bucket['reachable'] else None)
    return dict(bins=[dict(low=str(bins[i]), high=str(bins[i+1]), **record(b)) for i, b in enumerate(by_bin)],
                positions=[dict(position=i+1, **record(b)) for i, b in enumerate(positions)])


def access_groups(rows, reverse=False):
    for row in rows:
        if row['draft_routes_valid']:
            for stage in range(3):
                ids = list(dict.fromkeys(row['draft_routes'][stage*15:(stage+1)*15]))
                yield 'draft', [(40+stage)*384+e for e in (ids[::-1] if reverse else ids)]
        width = (row['k']+1)*6
        for layer in range(40):
            ids = list(dict.fromkeys(row['target_routes'][layer*width:(layer+1)*width]))
            yield 'main', [layer*384+e for e in (ids[::-1] if reverse else ids)]


def simulate(header, rows, mode, reverse=False):
    pinned = [(40+s)*384+e for s in range(3) for e in range(128)]
    dynamic = mode != 'pinned'
    capacity = header['slots'] if dynamic else header['slots']-header['draft_pins']
    cache = OrderedDict.fromkeys(header['initial_main_lru'] + (pinned if dynamic else []))
    if len(cache) > capacity or capacity <= 0:
        raise ValueError('initial residency exceeds cache capacity')
    stats = dict(main=Counter(), draft=Counter())
    for kind, keys in access_groups(rows, reverse):
        # Probe the entire union before filling, as mask stages a residency snapshot.
        misses = [key for key in keys if key not in cache] if kind == 'main' or dynamic else []
        stats[kind]['requests'] += len(keys)
        stats[kind]['misses'] += len(misses)
        for key in keys:
            if kind == 'draft' and not dynamic:
                continue
            if key in cache:
                cache.move_to_end(key)
        if kind == 'draft' and mode == 'draft-mask':
            # A cheap masked draft neither waits for nor reloads absent experts.
            continue
        protected = set(keys)
        for key in misses:
            if len(cache) == capacity:
                victim = next((x for x in cache if x not in protected), None)
                if victim is None:
                    raise ValueError('union does not fit in modeled cache')
                del cache[victim]
            cache[key] = None
            stats[kind]['fill_bytes'] += EXPERT_BYTES
    return dict(mode=mode, capacity=capacity, reverse_union_order=reverse,
                **{kind: dict(s, hit_rate=1-s['misses']/s['requests'] if s['requests'] else None) for kind, s in stats.items()})


def analyze(header, rows, accept_k):
    cost = {key: sum(r[key] for r in rows)/len(rows) for key in ('draft_ms','verify_ms','cpu_ms','commit_ms')}
    emitted = sum(r['emitted'] for r in rows)
    return dict(cycles=len(rows), emitted=emitted, drafted=sum(r['k'] for r in rows),
                accepted=sum(r['accepted'] for r in rows), accept_top_k=accept_k,
                mean_cycle=cost, confidence=confidence_report(rows, accept_k),
                lru=[simulate(header, rows, mode, rev) for mode in ('pinned','lru','draft-mask') for rev in (False, True)],
                limitations=[
                    'Small contiguous sampled trajectory; confidence is not calibrated on held-out data.',
                    'Ranks after the first rejection describe the original draft history, not an alternative path.',
                    'Synchronous LRU omits P0 completion timing, background filling, guards, and same-stamp slot ties.',
                    'Initial snapshot includes ready evictable main slots; in-flight fills and guarded slots are omitted.',
                    'Initial unpinned MTP experts are assumed newest; forward/reverse union orders bracket only order sensitivity.',
                    'Changing masks changes hidden states and later routes; this fixed-route simulation cannot predict MMLU or speed.',
                    'Draft confidence is available after draft execution; choosing k=0 then does not save the draft cost.'])


def self_test():
    h=dict(slots=400,draft_pins=384,initial_main_lru=list(range(16)))
    row=dict(k=1,position=10,emitted=2,accepted=1,confidence=[3.0],target_rank=[2],draft_routes=[0]*45,
             target_routes=[0]*480,draft_routes_valid=True,draft_ms=30.,verify_ms=160.,cpu_ms=.2,commit_ms=2.)
    a=analyze(h,[row],4)
    assert a['confidence']['positions'][0]['conditional_acceptance']==1
    assert a['lru'][0]['draft']['misses']==0
    assert a['lru'][0]['main']['misses']==39
    assert all(x['main']['requests']==40 for x in a['lru'])
    row['target_rank']=[4]
    assert confidence_report([row],4)['positions'][0]['conditional_acceptance']==0
    print('spec_diagnostics self-test passed')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('trace',nargs='?');ap.add_argument('--out');ap.add_argument('--accept-top-k',type=int,default=4)
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test();return
    if not args.trace or args.accept_top_k<1:ap.error('provide a trace and positive accept-top-k')
    h,r=read_trace(args.trace);text=json.dumps(analyze(h,r,args.accept_top_k),indent=2)+'\n'
    if args.out:Path(args.out).write_text(text)
    else:print(text,end='')


if __name__=='__main__':main()
