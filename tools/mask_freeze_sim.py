#!/usr/bin/env python3
"""Replay the proposed global mask freeze, using cache_sim.LRU; never uses a GPU.

Parquet traces carry gate weights. Native route.bin does not: for those traces
only the necessary churn gate is evaluated, never a fabricated weighted mass.
Frozen routes are replayed, so this does not predict output feedback or quality.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from cache_sim import LRU
from hitrate_sim import load_heat, REC


@dataclass
class Thresholds:
    enter: float = .95
    leave: float = .90
    layer_floor: float = .75
    hard_floor: float = .80
    warm_tokens: int = 32
    frozen_tokens: int = 16
    alpha: float = 1 / 8
    churn_window: int = 16
    churn_fraction: float = .01


class Controller:
    """Each observe is one committed token; reset at every new turn.

    In warm state the signal is PRE-admission mass: post-P0 computed mass would
    always be one, and would not test whether freezing this set is safe.
    """
    def __init__(self, capacity, thresholds=None, layers=40):
        self.capacity = capacity
        self.cfg = thresholds or Thresholds()
        self.layers = layers
        self.reset()

    def reset(self):
        self.frozen = False
        self.age = 0
        self.ewma = np.ones(self.layers, dtype=np.float64)
        self.churn = deque(maxlen=self.cfg.churn_window)
        self.tokens = deque(maxlen=32)

    def repeat(self, token):
        if token is None:
            return False
        self.tokens.append(int(token))
        t = list(self.tokens)
        if len(t) >= 4 and len(set(t[-4:])) == 1:
            return True
        if len(t) >= 9:
            tri = tuple(t[-3:])
            return sum(tuple(t[i:i+3]) == tri for i in range(len(t)-2)) >= 3
        return False

    def churn_ready(self):
        return (self.age >= self.cfg.warm_tokens
                and len(self.churn) == self.cfg.churn_window
                and sum(self.churn) < self.capacity * self.cfg.churn_fraction)

    def observe(self, mass, evictions, full, token=None):
        mass = np.asarray(mass, dtype=np.float64)
        if mass.shape != (self.layers,) or not np.isfinite(mass).all() or np.any((mass < 0) | (mass > 1 + 1e-6)):
            raise ValueError("invalid retained gate mass")
        self.ewma += self.cfg.alpha * (mass - self.ewma)
        self.churn.append(int(evictions))
        self.age += 1
        repeat = self.repeat(token)
        reason = None
        if self.frozen:
            if mass.mean() < self.cfg.hard_floor:
                reason = "hard_floor"
            elif repeat:
                reason = "repeat"
            elif self.age >= self.cfg.frozen_tokens:
                if self.ewma.mean() < self.cfg.leave:
                    reason = "mean_mass"
                elif self.ewma.min() < self.cfg.layer_floor:
                    reason = "layer_mass"
            if reason:
                self.frozen = False
                self.age = 0
                self.churn.clear()
        elif not repeat and full and self.churn_ready() and self.ewma.mean() >= self.cfg.enter:
            self.frozen = True
            self.age = 0
            reason = "freeze"
        return reason


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_run(folder, capacity, heat):
    folder = Path(folder)
    raw = (folder / "route.bin").read_bytes()
    if not raw or len(raw) % REC:
        raise ValueError("empty or partial route.bin")
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, REC)
    ids = a[:, 8:488].copy().view('<u2').reshape(-1, 40, 6)
    if np.any(ids >= 384):
        raise ValueError("invalid expert id")
    turns = json.loads((folder / "turns.json").read_text())["turns"]
    kinds = []
    for i, t in enumerate(turns):
        if t["prefill_mode"] == "decode":
            kinds += [(False, i)] * t["prefill_tokens"]
        kinds += [(True, i)] * t["decode_steps"]
    if len(kinds) != len(ids):
        raise ValueError(f"route/turn count mismatch: {len(ids)} vs {len(kinds)}")
    cache = LRU(capacity)
    for key in reversed(load_heat(heat)[:capacity]):
        cache.admit(key)
    ctl = Controller(capacity)
    last = None
    candidates = decoded = 0
    rolling = []
    for row, (decode, turn) in zip(ids, kinds):
        if turn != last:
            ctl.reset()
            last = turn
        evictions = 0
        for layer, experts in enumerate(row):
            for e in experts:
                key = layer * 384 + int(e)
                if cache.contains(key):
                    cache.touch(key)
                else:
                    evictions += cache.admit(key) is not None
        if decode:
            decoded += 1
            ctl.age += 1
            ctl.churn.append(evictions)
            if ctl.age >= ctl.cfg.warm_tokens:
                rolling.append(sum(ctl.churn))
                candidates += ctl.churn_ready()
    return dict(input=str(folder), sha256=digest(folder / "route.bin"), capacity=capacity,
                records=len(ids), decode_tokens=decoded, weighted_mass=None,
                churn_candidates=candidates,
                rolling16_min=min(rolling) if rolling else None,
                rolling16_median=float(np.median(rolling)) if rolling else None,
                rolling16_max=max(rolling) if rolling else None,
                limitation="Demand-LRU replay with immediate fills and static initial heat; GPU prefill routes, async IO, P3 and output feedback absent. Native trace has no gate weights or output ids.")


def weighted_trace(folder):
    import pyarrow.parquet as pq
    paths = sorted(Path(folder).glob("route_layer*.parquet"))
    if len(paths) != 40:
        raise ValueError("weighted trace must contain all 40 layers")
    tables = [pq.read_table(p, columns=["prompt_id", "pos", "layer", "top6_ids", "top6_weights"]) for p in paths]
    prompt = tables[0]["prompt_id"].to_numpy()
    pos = tables[0]["pos"].to_numpy()
    for layer, t in enumerate(tables):
        if not (np.array_equal(prompt, t["prompt_id"].to_numpy())
                and np.array_equal(pos, t["pos"].to_numpy())
                and np.all(t["layer"].to_numpy() == layer)):
            raise ValueError("layer/token alignment mismatch")
    def stack(column):
        return np.stack([t[column].combine_chunks().flatten().to_numpy().reshape(-1, 6) for t in tables], axis=1)
    ids, weights = stack("top6_ids"), stack("top6_weights")
    if np.any(ids >= 384) or not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights.sum(2) <= 0):
        raise ValueError("invalid weighted route")
    order = np.lexsort((pos, prompt))
    return prompt[order], ids[order], weights[order], {p.name: digest(p) for p in paths}


def weighted_run(data, capacity, enter, leave):
    prompts, ids, weights, _ = data
    cache = LRU(capacity)
    ctl = Controller(capacity, Thresholds(enter=enter, leave=leave))
    previous = None
    mass_sum = frozen_mass = 0.
    frozen = candidates = switches = 0
    reasons = {}
    for prompt, row, w in zip(prompts, ids, weights):
        if prompt != previous:
            ctl.reset()
            previous = prompt
        retained = np.zeros(40)
        evictions = 0
        was_frozen = ctl.frozen
        for layer in range(40):
            for e, z in zip(row[layer], w[layer]):
                key = layer * 384 + int(e)
                if cache.contains(key):
                    retained[layer] += z
                    if not was_frozen:
                        cache.touch(key)
                elif not was_frozen:
                    evictions += cache.admit(key) is not None
        retained /= w.sum(1)
        mean = float(retained.mean())
        mass_sum += mean
        if was_frozen:
            frozen += 1
            frozen_mass += mean
        reason = ctl.observe(retained, evictions, cache.size() == capacity)
        candidates += not was_frozen and (ctl.churn_ready() or reason == "freeze")
        if reason:
            switches += 1
            reasons[reason] = reasons.get(reason, 0) + 1
    f = frozen / len(ids)
    return dict(capacity=capacity, enter=enter, leave=leave, tokens=len(ids),
                frozen_tokens=frozen, frozen_fraction=f,
                mean_pre_admission_mass=mass_sum / len(ids),
                mean_frozen_mass=frozen_mass / frozen if frozen else None,
                switches=switches, switch_reasons=reasons, churn_candidates=candidates,
                zero_cost_frozen_token_speedup_ceiling=f / (1 - f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-run", action="append", default=[])
    p.add_argument("--weighted-trace")
    p.add_argument("--capacities", default="5500,5116")
    p.add_argument("--thresholds", default="0.93/0.88,0.95/0.90,0.97/0.92")
    p.add_argument("--heat", default="store/static_heat.inc")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    capacities = [int(x) for x in a.capacities.split(",")]
    thresholds = [tuple(map(float, x.split("/"))) for x in a.thresholds.split(",")]
    if any(c <= 0 for c in capacities) or any(not 0 < leave < enter <= 1 for enter, leave in thresholds):
        p.error("invalid capacities or thresholds")
    result = dict(controller=vars(Thresholds()), native=[], weighted=[], weighted_sources={})
    for folder in a.native_run:
        result["native"] += [native_run(folder, c, a.heat) for c in capacities]
    if a.weighted_trace:
        data = weighted_trace(a.weighted_trace)
        result["weighted_sources"] = data[3]
        result["weighted_limitation"] = "Recorded mixed teacher-forced prompt routes, not generated tokens. Warm signal is pre-admission resident mass; no P3 reheat, repetition ids, runtime fencing, IO cadence, MTP pin keys or feedback. 5116 approximates 5500 minus 384 pinned MTP slots. Speed ceiling assumes uniform token cost and zero total cost for frozen tokens; it is not measured throughput."
        for c in capacities:
            for enter, leave in thresholds:
                row = weighted_run(data, c, enter, leave)
                result["weighted"].append(row)
                print(json.dumps(row), flush=True)
    target = Path(a.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["native"], indent=2))


if __name__ == "__main__":
    main()
