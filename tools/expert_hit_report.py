#!/usr/bin/env python3
"""Per-layer expert-cache hit rates and per-task expert "partitions", from data on disk.

    .venv/bin/python tools/expert_hit_report.py \
        [--runs DIR ...] [--trace traces/mixed] [--out bench/results/linux/expert_hits] \
        [--caps 3000,3500,...,7000,5000,5499] [--switch-n 256] [--prewarm 500,1000,2000,3000]

Pure numpy over files already written; it never touches the GPU or the engine.

Inputs
------
* `--runs`: tools/hitrate_bench.py output dirs (route.bin, turns.json, profile.jsonl).
  route.bin (runtime/engine.h's dump): per step u32 step, u32 position, u16[40x6] expert
  ids in gate order, u8[40] MEASURED per-layer hit counts. The step layout is rebuilt from
  turns.json: a turn with prefill_mode "gpu" has NO prefill records (GPU batched prefill
  does not go through the planner's per-token path), a "decode" turn has prefill_tokens
  prefill records then decode_steps decode records. Records in front of the first turn
  (in the current runs: 128 = the bounded window replay of the parked session restored
  from kv_disk at startup) are labelled "startup": they touch the cache, so the LRU
  replay includes them, but no per-turn statistic does. The first run is the primary one.
* `--trace`: tools/route_trace.py output (route_layerNN.parquet: prompt_id, pos,
  top6_ids ...). Only prompt_id/pos/top6_ids are read. The prompt -> kind (code/en/zh)
  map is not in meta.json; it is read from the pickled prompt list inside state.pt
  (zip member data.pkl, unpickled with every tensor stubbed out, so nothing big is
  loaded) and cached as prompt_kinds.json in --out. `--kinds FILE` overrides it.

What it computes (all written to --out)
--------------------------------------
(1) per layer: measured decode/prefill hit rate, misses per decode token, share of all
    decode misses; the layer x turn matrix; the same accesses replayed through an exact
    global LRU at the run's capacity (per-layer sim vs measured, and the fraction of
    steps whose per-layer hit count matches exactly); routing concentration (distinct
    experts, entropy, effective experts, experts covering 50/80/95% of routing mass).
(2) per task (the chat turn labels, and code/en/zh from the trace): per-layer working
    set, top experts, pairwise Jaccard of the 80%-mass sets, weighted overlap and cosine
    of the frequency vectors, split-half noise baselines, task-specific experts
    (share ratio >= 3x vs the other tasks, with minimum support), and per-layer
    task-specificity (Jensen-Shannon between tasks minus within-task split-half).
(3) simulation on the trace, exact global LRU at every capacity at once (LRU stack
    distances): single-task streams, task switches (hit rate of the first N tokens of
    task B after a cache warmed on task A), mixed orders of the same tokens (trace
    order, blocked, prompt round-robin, token round-robin = 3 concurrent streams),
    and task-aware ideas: static per-task partitions, pinning a task's top experts,
    and pre-warming a task's top experts at a switch; plus the pre-warm replayed on the
    real chat run (profiles taken from the trace kinds). Hit changes are converted
    to tok/s with ms/token ~= 90 + 5.4 * misses/token (approximate).

The exact-LRU replay is the planner's demand path (tools/hitrate_sim.py Lru): for each
step, layers 0..39 in order, the six ids in gate order. Stack distances give the same
hit/miss for every capacity in one pass; the script checks this against Lru on the
primary run.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import pickle
import sys
import time
import zipfile

import numpy as np

sys.dont_write_bytecode = True
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import hitrate_sim  # noqa: E402

L, K, E = 40, 6, 384
NKEYS = L * E
BIG = 1 << 30
MS_COMPUTE = 90.0        # ms/token of compute (approximate, STATUS numbers)
MS_PER_MISS = 5.4        # ms of NVMe stall per decode miss (72 ms / 13.4 misses)
GROUPS = [("L0-1 (SWA only)", range(0, 2)), ("L2-19 (compress 2)", range(2, 20)),
          ("L20-39 (compress 1)", range(20, 40))]
DEFAULT_RUNS = [
    "bench/results/linux/reboot112/auto_ab/2_auto",
    "bench/results/linux/reboot112/auto_ab/4_auto",
    "bench/results/linux/reboot112/auto_ab/1_cap5000",
    "bench/results/linux/reboot112/auto_ab/3_cap5000",
    "bench/results/linux/pfmin/4_newdefault",
]
KINDS = ("code", "en", "zh")


def group_of(layer):
    for name, r in GROUPS:
        if layer in r:
            return name
    return "?"


def ms_token(miss_per_tok):
    return MS_COMPUTE + MS_PER_MISS * miss_per_tok


def tok_s(miss_per_tok):
    return 1000.0 / ms_token(miss_per_tok)


def label_kind(label):
    s = label.lower()
    if "code" in s:
        return "code"
    if s.startswith("zh"):
        return "zh"
    if s.startswith("en"):
        return "en"
    return None


# --------------------------------------------------------------------------- #
# exact LRU over all capacities: stack distances
# --------------------------------------------------------------------------- #

class LruStack:
    """Global LRU stack. access(keys) returns, per key, the LRU stack distance
    (number of distinct keys touched since this key's previous access; BIG if never
    seen). A key hits in an LRU of capacity C iff distance < C. Keys inside one
    access() call must be distinct (one step's 240 keys always are)."""

    def __init__(self, nkeys=NKEYS):
        self.ts = np.full(nkeys, -1, np.int64)
        self.srt = np.empty(0, np.int64)     # sorted timestamps of every resident-ever key
        self.clock = 0

    def copy(self):
        o = LruStack.__new__(LruStack)
        o.ts, o.srt, o.clock = self.ts.copy(), self.srt.copy(), self.clock
        return o

    def access(self, keys):
        keys = np.asarray(keys, np.int64)
        n = len(keys)
        prev = self.ts[keys]
        seen = prev >= 0
        loc = np.searchsorted(self.srt, prev[seen])
        d = np.full(n, BIG, np.int64)
        # keys more recent than this key's previous access
        d[seen] = len(self.srt) - loc - 1
        # plus keys earlier in this call whose previous access was older (or never)
        if n > 1:
            before = np.tril(prev[None, :] <= prev[:, None], -1).sum(1)
            d[seen] += before[seen]
        self.srt = np.concatenate([np.delete(self.srt, loc), self.clock + np.arange(n)])
        self.ts[keys] = self.clock + np.arange(n)
        self.clock += n
        return d

    def stream(self, keys2d):
        """keys2d: [T, 240] -> distances [T, 240]."""
        out = np.empty(keys2d.shape, np.int64)
        for t in range(keys2d.shape[0]):
            out[t] = self.access(keys2d[t])
        return out


def to_keys(ids):
    """ids [T, 40, 6] -> global keys [T, 240] in layer-major, gate order."""
    return (ids.astype(np.int64) + (np.arange(L, dtype=np.int64) * E)[None, :, None]).reshape(len(ids), L * K)


def hits_by_layer(dist, cap):
    """dist [T, 240] -> hits [T, 40]."""
    return (dist < cap).reshape(len(dist), L, K).sum(2)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_run(d):
    step, pos, ids, hits = hitrate_sim.load_route(os.path.join(d, "route.bin"))
    tj = json.load(open(os.path.join(d, "turns.json"), encoding="utf-8"))
    turns = tj["turns"]
    layout = []
    for i, t in enumerate(turns):
        if t.get("prefill_mode", "decode") != "gpu":
            layout += [("prefill", i)] * int(t.get("prefill_tokens", 0))
        layout += [("decode", i)] * int(t.get("decode_steps", 0))
    n = len(ids)
    lead = n - len(layout)
    notes = []
    if lead < 0:
        notes.append(f"route.bin has {n} records but turns.json implies {len(layout)}; truncated")
        layout = layout[:n]
        lead = 0
    kinds = ["startup"] * lead + [k for k, _ in layout]
    turn = np.array([-1] * lead + [i for _, i in layout], np.int32)
    kind = np.array(kinds)
    # sanity check on positions: first record of each turn
    bad = 0
    for i, t in enumerate(turns):
        idx = np.where(turn == i)[0]
        if not len(idx):
            continue
        if t.get("prefill_mode", "decode") == "gpu":
            want = int(t["prompt_tokens"])
        else:
            want = int(t["prompt_tokens"]) - int(t.get("prefill_tokens", 0))
        if int(pos[idx[0]]) != want:
            bad += 1
    if bad:
        notes.append(f"{bad} turns whose first record position does not match turns.json")
    if lead:
        notes.append(f"{lead} leading records before turn 1 (positions {pos[0]}..{pos[lead - 1]}), "
                     "labelled startup")
    prof = []
    pp = os.path.join(d, "profile.jsonl")
    if os.path.exists(pp):
        prof = [json.loads(l) for l in open(pp, encoding="utf-8") if l.strip()]
    srv = tj.get("server", {})
    name = os.path.relpath(os.path.abspath(d), REPO)
    return dict(name=name, short=name.replace("bench/results/linux/", "").replace("/", "_"),
                pos=pos, ids=ids, hits=hits, kind=kind, turn=turn, turns=turns,
                labels=[t.get("label", f"turn{i}") for i, t in enumerate(turns)],
                slots=int(srv.get("cache_slots", 0)), gpu_prefill_min=srv.get("gpu_prefill_min"),
                prefill_modes=[t.get("prefill_mode", "decode") for t in turns],
                prof=prof, notes=notes)


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return None

    def __setstate__(self, s):
        pass


class _SafeUnpickler(pickle.Unpickler):
    """Reads the plain-Python part of a torch.save zip; every tensor becomes None."""

    def find_class(self, mod, name):
        if mod == "collections":
            return getattr(collections, name)
        if mod == "builtins" and name in ("list", "dict", "set", "tuple", "str", "int", "float"):
            return getattr(__import__("builtins"), name)
        if mod.startswith("torch"):
            return lambda *a, **k: None
        return _Stub

    def persistent_load(self, pid):
        return None


def prompt_kinds(trace_dir, out_dir, override):
    if override:
        m = json.load(open(override, encoding="utf-8"))
        return {int(k): v for k, v in m["kinds"].items()}, f"from {override}"
    cache = os.path.join(out_dir, "prompt_kinds.json")
    sp = os.path.join(trace_dir, "state.pt")
    if os.path.exists(sp):
        z = zipfile.ZipFile(sp)
        pk = [n for n in z.namelist() if n.endswith("data.pkl")]
        blob = _SafeUnpickler(z.open(pk[0])).load()
        prompts = blob["prompts"]
        kinds = {i: p["kind"] for i, p in enumerate(prompts)}
        info = {i: {"kind": p["kind"], "source": p.get("source", ""),
                    "n_ids": len(p["ids"]) if isinstance(p.get("ids"), list) else None,
                    "head": (p.get("text") or "")[:60]} for i, p in enumerate(prompts)}
        with open(cache, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"source": "state.pt (route_trace.py checkpoint, prompts list)",
                       "kinds": kinds, "prompts": info}, f, ensure_ascii=False, indent=1)
        return kinds, "exact, from state.pt's pickled prompt list"
    if os.path.exists(cache):
        m = json.load(open(cache, encoding="utf-8"))
        return {int(k): v for k, v in m["kinds"].items()}, f"from cached {cache}"
    raise SystemExit("no prompt->kind map: state.pt missing and no --kinds given")


def load_trace(tdir):
    import pyarrow.parquet as pq
    ids, pid0, pos0 = None, None, None
    for l in range(L):
        t = pq.read_table(os.path.join(tdir, f"route_layer{l:02d}.parquet"),
                          columns=["prompt_id", "pos", "top6_ids"])
        pid = t.column("prompt_id").to_numpy().astype(np.int32)
        pos = t.column("pos").to_numpy().astype(np.int64)
        top = t.column("top6_ids").combine_chunks().flatten().to_numpy().astype(np.uint16).reshape(-1, K)
        order = np.lexsort((pos, pid))
        pid, pos, top = pid[order], pos[order], top[order]
        if ids is None:
            ids = np.empty((len(pid), L, K), np.uint16)
            pid0, pos0 = pid, pos
        elif not (np.array_equal(pid, pid0) and np.array_equal(pos, pos0)):
            raise SystemExit(f"layer {l} rows do not line up with layer 0")
        ids[:, l, :] = top
    return pid0, pos0, ids


# --------------------------------------------------------------------------- #
# statistics helpers
# --------------------------------------------------------------------------- #

def counts_of(ids):
    """ids [T, 40, 6] -> [40, 384] routing counts."""
    c = np.zeros((L, E), np.int64)
    for l in range(L):
        c[l] = np.bincount(ids[:, l, :].ravel(), minlength=E)
    return c


def conc_row(c):
    tot = c.sum()
    if tot == 0:
        return dict(distinct=0, entropy_bits=0.0, eff_experts=0.0, n50=0, n80=0, n95=0, top1=0.0, top16=0.0)
    p = c / tot
    nz = p[p > 0]
    h = float(-(nz * np.log2(nz)).sum())
    s = np.sort(p)[::-1]
    cum = np.cumsum(s)
    return dict(distinct=int((c > 0).sum()), entropy_bits=h, eff_experts=2 ** h,
                n50=int(np.searchsorted(cum, 0.5 - 1e-12) + 1),
                n80=int(np.searchsorted(cum, 0.8 - 1e-12) + 1),
                n95=int(np.searchsorted(cum, 0.95 - 1e-12) + 1),
                top1=float(s[0]), top16=float(s[:16].sum()))


def mass_set(c, q=0.8):
    tot = c.sum()
    if tot == 0:
        return set()
    o = np.argsort(-c, kind="stable")
    cum = np.cumsum(c[o]) / tot
    n = int(np.searchsorted(cum, q - 1e-12) + 1)
    return set(o[:n].tolist())


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0


def wov(p, q):
    return float(np.minimum(p, q).sum())


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def jsd(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / b[nz])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def norm(c):
    s = c.sum()
    return c / s if s else c.astype(float)


def mean_distinct_window(ids, w):
    """Mean distinct experts per layer in non-overlapping w-token windows -> [40]."""
    T = len(ids)
    if T == 0:
        return np.zeros(L)
    starts = list(range(0, max(1, T - w + 1), w)) if T >= w else [0]
    out = np.zeros(L)
    for s in starts:
        blk = ids[s:s + w]
        for l in range(L):
            out[l] += len(np.unique(blk[:, l, :]))
    return out / len(starts)


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})


def r4(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(float(x)) else round(float(x), 5)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, dict):
        return {str(k): r4(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r4(v) for v in x]
    if isinstance(x, np.ndarray):
        return r4(x.tolist())
    return x


# --------------------------------------------------------------------------- #
# part 1: measured per-layer hit rates
# --------------------------------------------------------------------------- #

def analyse_run(run, caps, selfcheck):
    ids, hits, kind, turn = run["ids"], run["hits"], run["kind"], run["turn"]
    dec, pre = kind == "decode", kind == "prefill"
    nd = int(dec.sum())
    keys = to_keys(ids)
    t0 = time.time()
    dist = LruStack().stream(keys)
    sim_caps = sorted(set([run["slots"]] + list(caps)))
    sim = {c: hits_by_layer(dist, c) for c in sim_caps}
    check = None
    if selfcheck:
        ref = hitrate_sim.simulate(ids, run["slots"])
        check = bool(np.array_equal(ref, sim[run["slots"]]))
    mine = sim[run["slots"]]
    miss_dec = (K - hits[dec]).sum(0)             # [40]
    tot_miss = miss_dec.sum()
    rows = []
    for l in range(L):
        rows.append(dict(
            layer=l, group=group_of(l),
            decode_hit=hits[dec, l].sum() / (K * nd) if nd else float("nan"),
            prefill_hit=hits[pre, l].sum() / (K * pre.sum()) if pre.sum() else float("nan"),
            decode_miss_per_tok=miss_dec[l] / nd if nd else float("nan"),
            decode_miss_share=miss_dec[l] / tot_miss if tot_miss else float("nan"),
            sim_decode_hit=mine[dec, l].sum() / (K * nd) if nd else float("nan"),
            sim_prefill_hit=mine[pre, l].sum() / (K * pre.sum()) if pre.sum() else float("nan"),
            step_agree=float((mine[:, l] == hits[:, l]).mean()),
            **{f"sim{c}_decode_hit": sim[c][dec, l].sum() / (K * nd) for c in sim_caps if c != run["slots"]},
        ))
    # layer x turn (decode) matrix
    lt = np.full((L, len(run["turns"])), np.nan)
    for i in range(len(run["turns"])):
        m = dec & (turn == i)
        if m.sum():
            lt[:, i] = hits[m].sum(0) / (K * m.sum())
    # turn starts: first 32 / 64 decode steps vs the rest, and the first steps of the turn
    starts = []
    for i, lab in enumerate(run["labels"]):
        di = np.where(dec & (turn == i))[0]
        ai = np.where(turn == i)[0]
        if not len(di):
            continue
        r = dict(turn=i, label=lab, prefill_mode=run["prefill_modes"][i],
                 new_topic=bool(run["turns"][i].get("reused_tokens", 0) == 0),
                 prefill_steps=int((pre & (turn == i)).sum()), decode_steps=len(di))
        for n in (16, 32, 64):
            r[f"first{n}_step_hit"] = hits[ai[:n]].sum() / (K * L * len(ai[:n]))
            r[f"first{n}_decode_hit"] = hits[di[:n]].sum() / (K * L * len(di[:n]))
        rest = di[64:]
        r["rest_decode_hit"] = hits[rest].sum() / (K * L * len(rest)) if len(rest) else float("nan")
        r["turn_decode_hit"] = hits[di].sum() / (K * L * len(di))
        r["sim_first64_decode_hit"] = mine[di[:64]].sum() / (K * L * len(di[:64]))
        r["sim_turn_decode_hit"] = mine[di].sum() / (K * L * len(di))
        starts.append(r)
    tot = dict(
        records=len(ids), startup=int((kind == "startup").sum()), prefill=int(pre.sum()), decode=nd,
        slots=run["slots"], gpu_prefill_min=run["gpu_prefill_min"],
        decode_hit=hits[dec].sum() / (K * L * nd) if nd else float("nan"),
        prefill_hit=hits[pre].sum() / (K * L * pre.sum()) if pre.sum() else float("nan"),
        decode_miss_per_tok=(K * L - hits[dec].sum(1)).mean() if nd else float("nan"),
        sim_decode_hit=mine[dec].sum() / (K * L * nd) if nd else float("nan"),
        steps_exact=float((mine == hits).all(1).mean()),
        sim_equals_hitrate_sim_Lru=check,
        sim_decode_hit_by_cap={c: sim[c][dec].sum() / (K * L * nd) for c in sim_caps},
        notes=run["notes"], sim_seconds=round(time.time() - t0, 1),
    )
    if run["prof"] and len(run["prof"]) == len(ids):
        pr = run["prof"]
        mb = sum(p.get("miss_bytes", 0) for p in pr)
        mm = sum(p.get("expert_misses", 0) for p in pr)
        tot["bytes_per_miss"] = mb / mm if mm else None
        di = np.where(dec)[0]
        tot["measured_ms_token"] = float(np.mean([pr[i]["wall_ms"] for i in di]))
        tot["measured_stall_ms"] = float(np.mean([pr[i]["nvme_stall_ms"] for i in di]))
        tot["measured_tok_s"] = 1000.0 / tot["measured_ms_token"]
    return rows, lt, starts, tot


# --------------------------------------------------------------------------- #
# part 2: task partitions
# --------------------------------------------------------------------------- #

def task_tables(src, tasks, split_halves, min_count, ratio_min):
    """tasks: {name: ids[T,40,6]}; split_halves: {name: (idsA, idsB)} for the noise
    baseline. Returns rows for the CSVs plus per-layer summaries."""
    names = list(tasks)
    cnt = {n: counts_of(tasks[n]) for n in names}
    ntok = {n: len(tasks[n]) for n in names}
    stat_rows, top_rows, spec_rows, ov_rows = [], [], [], []
    wd = {n: mean_distinct_window(tasks[n], 128) for n in names}
    for n in names:
        for l in range(L):
            c = cnt[n][l]
            cr = conc_row(c)
            o = np.argsort(-c, kind="stable")[:8]
            stat_rows.append(dict(source=src, task=n, layer=l, tokens=ntok[n], **cr,
                                  distinct_per_128tok=wd[n][l],
                                  top8=";".join(f"{e}:{c[e] / c.sum():.3f}" for e in o)))
    # pairwise overlap
    pair_avg = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            acc = []
            for l in range(L):
                pa, pb = norm(cnt[a][l]), norm(cnt[b][l])
                row = dict(source=src, a=a, b=b, layer=l,
                           jaccard80=jaccard(mass_set(cnt[a][l]), mass_set(cnt[b][l])),
                           weighted_overlap=wov(pa, pb), cosine=cosine(pa, pb), jsd=jsd(pa, pb))
                ov_rows.append(row)
                acc.append(row)
            pair_avg[(a, b)] = {k: float(np.mean([r[k] for r in acc]))
                                for k in ("jaccard80", "weighted_overlap", "cosine", "jsd")}
    # split-half (same task) baselines
    self_avg = {}
    for n, (ha, hb) in split_halves.items():
        ca, cb = counts_of(ha), counts_of(hb)
        acc = []
        for l in range(L):
            pa, pb = norm(ca[l]), norm(cb[l])
            row = dict(source=src, a=n + "#A", b=n + "#B", layer=l,
                       jaccard80=jaccard(mass_set(ca[l]), mass_set(cb[l])),
                       weighted_overlap=wov(pa, pb), cosine=cosine(pa, pb), jsd=jsd(pa, pb))
            ov_rows.append(row)
            acc.append(row)
        self_avg[n] = {k: float(np.mean([r[k] for r in acc]))
                       for k in ("jaccard80", "weighted_overlap", "cosine", "jsd")}
        self_avg[n]["jsd_by_layer"] = [r["jsd"] for r in acc]
    # task-specific experts: share in the task vs share in all other tasks together
    total = sum(cnt.values())
    for n in names:
        other = total - cnt[n]
        nt, no = ntok[n] * K, (sum(ntok.values()) - ntok[n]) * K
        for l in range(L):
            st = cnt[n][l] / nt
            so = other[l] / no
            floor = 1.0 / no                      # one routing in the others
            ratio = st / np.maximum(so, floor)
            ok = (cnt[n][l] >= min_count) & (ratio >= ratio_min) & (st >= 1.0 / E)
            for e in np.where(ok)[0]:
                spec_rows.append(dict(source=src, task=n, layer=l, expert=int(e),
                                      count=int(cnt[n][l, e]), share_task=float(st[e]),
                                      share_others=float(so[e]), ratio=float(ratio[e]),
                                      others_zero=bool(other[l, e] == 0)))
    # share of each task's routing mass that lands on its own specific experts
    spec_mass = {}
    for n in names:
        m = np.zeros((L, E), bool)
        for r in spec_rows:
            if r["task"] == n:
                m[r["layer"], r["expert"]] = True
        spec_mass[n] = {gname: float(cnt[n][list(rg)][m[list(rg)]].sum() / cnt[n][list(rg)].sum())
                        for gname, rg in GROUPS}
        spec_mass[n]["all"] = float(cnt[n][m].sum() / cnt[n].sum())
    # per-layer task specificity: mean pairwise JSD between tasks minus mean within-task JSD
    spec_layer = []
    for l in range(L):
        between = [jsd(norm(cnt[a][l]), norm(cnt[b][l])) for i, a in enumerate(names) for b in names[i + 1:]]
        within = [self_avg[n]["jsd_by_layer"][l] for n in self_avg]
        sets = [mass_set(cnt[n][l]) for n in names]
        core = set.intersection(*sets) if sets else set()
        union = set.union(*sets) if sets else set()
        allc = sum(cnt[n][l] for n in names)
        spec_layer.append(dict(source=src, layer=l, group=group_of(l),
                               jsd_between=float(np.mean(between)),
                               jsd_within=float(np.mean(within)) if within else float("nan"),
                               jsd_excess=float(np.mean(between) - (np.mean(within) if within else 0)),
                               core80_size=len(core), union80_size=len(union),
                               core80_mass=float(allc[list(core)].sum() / allc.sum()) if core else 0.0,
                               n_specific=sum(1 for r in spec_rows if r["layer"] == l)))
    for s in self_avg.values():
        s.pop("jsd_by_layer", None)
    # global footprints: (layer, expert) keys covering q of the task's routing mass
    foot = {}
    for n in names:
        flat = cnt[n].ravel()
        o = np.sort(flat)[::-1]
        cum = np.cumsum(o) / o.sum()
        foot[n] = {f"keys{int(q * 100)}": int(np.searchsorted(cum, q - 1e-12) + 1) for q in (0.5, 0.8, 0.9, 0.95)}
        foot[n]["distinct_keys"] = int((flat > 0).sum())
    unions = {}
    for q in (0.8, 0.9):
        sets = []
        for n in names:
            flat = cnt[n].ravel()
            o = np.argsort(-flat, kind="stable")
            cum = np.cumsum(flat[o]) / flat.sum()
            sets.append(set(o[:int(np.searchsorted(cum, q - 1e-12) + 1)].tolist()))
        unions[f"union{int(q * 100)}"] = len(set.union(*sets))
        unions[f"intersection{int(q * 100)}"] = len(set.intersection(*sets))
    return dict(cnt=cnt, ntok=ntok, stat_rows=stat_rows, spec_rows=spec_rows, ov_rows=ov_rows,
                spec_mass=spec_mass, pair_avg=pair_avg, self_avg=self_avg, spec_layer=spec_layer, footprint=foot,
                unions=unions)


def replication(kind_prompts, pid_ids, min_count, ratio_min):
    """Specific experts found on the even prompts of each kind (vs all other kinds),
    re-tested on the odd prompts: the fraction that is still >= 2x there."""
    halves = {}
    for k, pl in kind_prompts.items():
        halves[k] = (pl[0::2], pl[1::2])
    res = {}
    for k in kind_prompts:
        ev = np.concatenate([pid_ids[p] for p in halves[k][0]])
        od = np.concatenate([pid_ids[p] for p in halves[k][1]])
        ev_o = np.concatenate([pid_ids[p] for kk in kind_prompts if kk != k for p in halves[kk][0]])
        od_o = np.concatenate([pid_ids[p] for kk in kind_prompts if kk != k for p in halves[kk][1]])
        ce, co, cee, coo = counts_of(ev), counts_of(od), counts_of(ev_o), counts_of(od_o)
        se, so_ = ce / (len(ev) * K), cee / (len(ev_o) * K)
        re_ = se / np.maximum(so_, 1.0 / (len(ev_o) * K))
        found = (ce >= min_count) & (re_ >= ratio_min) & (se >= 1.0 / E)
        s2, so2 = co / (len(od) * K), coo / (len(od_o) * K)
        r2 = s2 / np.maximum(so2, 1.0 / (len(od_o) * K))
        nf = int(found.sum())
        res[k] = dict(found_on_even=nf, still_2x_on_odd=int((found & (r2 >= 2)).sum()),
                      still_3x_on_odd=int((found & (r2 >= 3)).sum()),
                      frac_2x=float((found & (r2 >= 2)).sum() / nf) if nf else float("nan"))
    return res


# --------------------------------------------------------------------------- #
# part 3: simulation on the trace
# --------------------------------------------------------------------------- #

def curve(dist, caps, ntok):
    """dist [T, 240] -> {cap: hit rate}, {cap: misses/token}."""
    h = {c: float((dist < c).mean()) for c in caps}
    m = {c: float((dist >= c).sum() / ntok) for c in caps}
    return h, m


def top_keys(cnt, m):
    """Top-m (layer, expert) keys by count, hottest first."""
    flat = cnt.ravel()
    o = np.argsort(-flat, kind="stable")
    o = o[flat[o] > 0]
    return o[:m]


def prewarm(state, keys_hot_first):
    """Insert keys coldest first so the hottest is most recent. Returns distances."""
    ks = np.asarray(keys_hot_first[::-1], np.int64)
    d = np.empty(len(ks), np.int64)
    for s in range(0, len(ks), 512):
        d[s:s + 512] = state.access(ks[s:s + 512])
    return d


def pinned_stream(keys2d, pinned_mask):
    """Plain LRU on the accesses that are not pinned. Returns ([T] pinned hits,
    list of per-token distances of the unpinned accesses)."""
    st = LruStack()
    ph = np.zeros(len(keys2d), np.int64)
    dl = []
    for t in range(len(keys2d)):
        k = keys2d[t]
        pm = pinned_mask[k]
        ph[t] = pm.sum()
        dl.append(st.access(k[~pm]))
    return ph, dl


def simulate_trace(pid, ids, kinds, caps, switch_n, prewarm_ms, pin_ps, log):
    keys = to_keys(ids)
    prompts = list(dict.fromkeys(pid.tolist()))          # trace order
    rows_of = {p: np.where(pid == p)[0] for p in prompts}
    kp = {k: [p for p in prompts if kinds[p] == k] for k in KINDS}
    out = {"caps": caps}

    def cat(pl):
        return np.concatenate([rows_of[p] for p in pl]) if pl else np.empty(0, np.int64)

    # --- single-task streams (cold start, all prompts of the kind in trace order)
    single = {}
    single_dist = {}
    for k in KINDS:
        r = cat(kp[k])
        d = LruStack().stream(keys[r])
        single_dist[k] = d
        h, m = curve(d, caps, len(r))
        hw, mw = curve(d[1000:], caps, len(r) - 1000)
        single[k] = dict(tokens=len(r), hit=h, miss_per_tok=m, hit_after1000=hw, miss_after1000=mw,
                         per_layer_hit={c: ((d < c).reshape(len(r), L, K).mean((0, 2))).tolist()
                                        for c in (5000, 5499) if c in caps})
        log(f"  single {k}: {len(r)} tokens")
    out["single"] = single

    # --- orders of the same 27k tokens
    order_idx = {
        "trace_order": cat(prompts),
        "blocked_code_en_zh": cat(kp["code"] + kp["en"] + kp["zh"]),
        "prompt_round_robin": cat([p for grp in zip(*[kp[k] for k in KINDS]) for p in grp] +
                                  [p for k in KINDS for p in kp[k][min(len(kp[kk]) for kk in KINDS):]]),
    }
    streams = [cat(kp[k]) for k in KINDS]
    rr = []
    for t in range(max(len(s) for s in streams)):
        for s in streams:
            if t < len(s):
                rr.append(s[t])
    order_idx["token_round_robin_3streams"] = np.array(rr)
    orders = {}
    for name, idx in order_idx.items():
        assert len(idx) == len(pid)
        d = LruStack().stream(keys[idx])
        h, m = curve(d, caps, len(idx))
        hw, mw = curve(d[1000:], caps, len(idx) - 1000)
        orders[name] = dict(hit=h, miss_per_tok=m, hit_after1000=hw, miss_after1000=mw)
        if name == "token_round_robin_3streams":
            # static partitions: each task gets its own LRU pool (single-stream distances)
            part = {}
            for c in caps:
                eq = sum(float((single_dist[k] < c // 3).sum()) for k in KINDS)
                wset = {k: len(np.unique(keys[cat(kp[k])])) for k in KINDS}
                ws = sum(wset.values())
                pr = sum(float((single_dist[k] < int(c * wset[k] / ws)).sum()) for k in KINDS)
                tokp = {k: len(cat(kp[k])) for k in KINDS}
                prop_tok = sum(float((single_dist[k] < int(c * tokp[k] / len(pid))).sum()) for k in KINDS)
                part[c] = dict(shared_lru=h[c], equal_thirds=eq / (len(pid) * L * K),
                               by_working_set=pr / (len(pid) * L * K),
                               by_token_share=prop_tok / (len(pid) * L * K))
            orders[name]["partitions"] = part
        log(f"  order {name}")
    out["orders"] = orders

    # --- switches: warm on A's even prompts, probe B's odd prompts; + pre-warm B's top keys
    half_w = {k: kp[k][0::2] for k in KINDS}
    half_p = {k: kp[k][1::2] for k in KINDS}
    prof = {k: counts_of(ids[cat(half_w[k])]) for k in KINDS}
    prof_all = sum(prof.values())
    warm_state = {}
    for a in KINDS:
        st = LruStack()
        st.stream(keys[cat(half_w[a])])
        warm_state[a] = st
    warm_state["cold"] = LruStack()
    windows = [(0, 16), (0, 64), (0, switch_n), (switch_n, 2 * switch_n)]
    sw = {}
    pw = {}
    probes = {}
    for b in KINDS:
        pl = half_p[b]
        probes[b] = []
        for j in range(min(4, len(pl))):
            seq = pl[j:] + pl[:j]
            r = cat(seq)[:2 * switch_n]
            probes[b].append(r)
    for a in list(KINDS) + ["cold"]:
        for b in KINDS:
            res = {}
            per = {c: np.zeros(2 * switch_n) for c in caps}
            npro = 0
            for r in probes[b]:
                st = warm_state[a].copy()
                d = st.stream(keys[r])
                for c in caps:
                    per[c][:len(r)] += (d < c).sum(1)
                npro += 1
            for c in caps:
                res[c] = {f"{w0}-{w1}": float(per[c][w0:w1].sum() / (npro * (w1 - w0) * L * K)) for w0, w1 in windows}
            sw[f"{a}->{b}"] = res
            # pre-warm variants (only real switches and cold)
            for mname, pk in (("task", prof[b]), ("global", prof_all)):
                for M in prewarm_ms:
                    hk = top_keys(pk, M)
                    per = {c: np.zeros(2 * switch_n) for c in caps}
                    loads = {c: 0.0 for c in caps}
                    for r in probes[b]:
                        st = warm_state[a].copy()
                        dp = prewarm(st, hk)
                        d = st.stream(keys[r])
                        for c in caps:
                            per[c][:len(r)] += (d < c).sum(1)
                            loads[c] += float((dp >= c).sum())
                    res2 = {}
                    for c in caps:
                        res2[c] = {f"{w0}-{w1}": float(per[c][w0:w1].sum() / (npro * (w1 - w0) * L * K))
                                   for w0, w1 in windows}
                        res2[c]["prewarm_loads"] = loads[c] / npro
                    pw[f"{a}->{b}|{mname}|{M}"] = res2
        log(f"  switches from {a}")
    out["switch"] = sw
    out["prewarm"] = pw
    out["switch_windows"] = [f"{w0}-{w1}" for w0, w1 in windows]

    # --- pinning a task's top-P keys + LRU on the rest (probe half of B, cold start)
    pin = {}
    for b in KINDS:
        r = cat(half_p[b])
        base = LruStack().stream(keys[r])
        for mname, pk in (("task", prof[b]), ("global", prof_all)):
            for P in pin_ps:
                mask = np.zeros(NKEYS, bool)
                mask[top_keys(pk, P)] = True
                ph, dl = pinned_stream(keys[r], mask)
                dflat = np.concatenate(dl)
                for c in caps:
                    if c <= P:
                        continue
                    hit = (ph.sum() + (dflat < c - P).sum()) / (len(r) * L * K)
                    pin.setdefault(f"{b}|{mname}|{P}", {})[c] = dict(
                        hit=float(hit), lru_hit=float((base < c).mean()),
                        hit_after1000=float((ph[1000:].sum() +
                                             sum((x < c - P).sum() for x in dl[1000:])) /
                                            ((len(r) - 1000) * L * K)),
                        lru_hit_after1000=float((base[1000:] < c).mean()))
        log(f"  pinning {b}")
    out["pin"] = pin
    out["halves"] = {k: dict(warm=half_w[k], probe=half_p[k]) for k in KINDS}
    return out, kp


def chat_prewarm(run, trace_cnt_by_kind, ms, caps):
    """Replay the chat run through the exact LRU; at the first record of every
    new-topic turn, insert the top-M keys of that turn's kind (trace profile)."""
    keys = to_keys(run["ids"])
    kind, turn = run["kind"], run["turn"]
    dec = kind == "decode"
    first_of = {}
    for i, t in enumerate(run["turns"]):
        idx = np.where(turn == i)[0]
        if len(idx) and t.get("reused_tokens", 0) == 0 and label_kind(run["labels"][i]):
            first_of[int(idx[0])] = label_kind(run["labels"][i])
    first64 = np.zeros(len(keys), bool)
    for i in range(len(run["turns"])):
        if run["turns"][i].get("reused_tokens", 0) == 0:
            di = np.where(dec & (turn == i))[0][:64]
            first64[di] = True
    res = {}
    for M in [0] + list(ms):
        st = LruStack()
        dist = np.empty(keys.shape, np.int64)
        loads = {c: 0 for c in caps}
        for s in range(len(keys)):
            if M and s in first_of:
                dp = prewarm(st, top_keys(trace_cnt_by_kind[first_of[s]], M))
                for c in caps:
                    loads[c] += int((dp >= c).sum())
            dist[s] = st.access(keys[s])
        r = {}
        for c in caps:
            h = dist < c
            r[c] = dict(decode_hit=float(h[dec].mean()),
                        decode_miss_per_tok=float((~h[dec]).sum() / dec.sum()),
                        newtopic_first64_decode_hit=float(h[first64].mean()),
                        prefill_hit=float(h[kind == "prefill"].mean()) if (kind == "prefill").any() else None,
                        prewarm_loads_total=loads[c], prewarm_events=len(first_of) if M else 0)
        res[M] = r
    return res


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}%"


def md_table(hdr, rows):
    s = "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(c) for c in r) + " |\n"
    return s


def write_report(path, S):
    P = S["runs"][0]
    lay = P["layers"]
    T = S["trace"]
    tp = S["tasks"]
    sim = S["sim"]
    lines = []
    w = lines.append
    w("# 专家缓存逐层命中率与任务分区分析\n\n")
    w(f"由 `tools/expert_hit_report.py` 生成（{S['generated']}）。纯 CPU/numpy 离线分析，未碰 GPU。"
      "所有「模拟」都是精确全局 LRU（与 `tools/hitrate_sim.py` 的 `Lru` / 引擎 planner 的 demand 路径一致），"
      "用 LRU 栈距离一次得到所有容量的结果。tok/s 换算一律用近似模型 "
      f"`ms/token ≈ {MS_COMPUTE:.0f} + {MS_PER_MISS} × 每 token miss 数`，只作量级参考。\n")
    w("\n## 要点\n\n")
    for x in S["conclusions"]:
        w(f"- {x}\n")
    w("\n## 0. 数据\n\n")
    rows = []
    for R in S["runs"]:
        t = R["total"]
        rows.append([R["name"], t["slots"], t["startup"], t["prefill"], t["decode"], pct(t["decode_hit"]),
                     f"{t['decode_miss_per_tok']:.2f}", pct(t["sim_decode_hit"]), pct(t["steps_exact"]),
                     f"{t.get('measured_tok_s', float('nan')):.2f}"])
    w(md_table(["run", "槽位", "startup 步", "prefill 步", "decode 步", "实测 decode 命中", "miss/token",
                "LRU 复算命中", "逐步逐层完全一致", "实测 tok/s"], rows))
    w("\n说明：\n")
    w("- 四个 reboot112/auto_ab 运行的路由**逐 token 完全相同**（固定 seed），只是容量不同（5,000 / 5,400）。"
      "所以它们是同一条路由流的两个容量点，不是四个独立样本。任务里提到的 5,499 槽是当前 auto 的值，"
      "这批运行实际是 **5,400 槽**；5,499 只出现在模拟里。\n")
    w("- route.bin 在第 1 轮之前多出 128 条记录（位置 268..395），是启动时从 kv_disk 恢复上次 parked 会话的 "
      "128 步窗口重放。它们确实访问了缓存，所以 LRU 复算包含它们，但不计入任何轮次统计。"
      "**注意：`tools/hitrate_sim.py curve` 和 `tools/expert_patterns.py` 用 `step_kinds()` 从第 0 条开始对齐，"
      "对这些运行会把 decode/prefill 标签整体错位 128 步。**\n")
    w("- pfmin/4_newdefault 的新话题轮次走 GPU prefill，route.bin 里没有这些 prefill 步；GPU prefill 会把专家"
      "交接进缓存（planner 的 prefill handoff），这些插入不在 route.bin 里，所以这条运行的 LRU 复算必然偏离实测，"
      "只做参考。\n")
    w(f"- traces/mixed：{T['tokens']} token / {T['prompts']} 个 prompt，teacher-forced（整段 prompt 文本逐 token 路由）。"
      f"prompt→kind 映射：{T['kind_source']}。按 kind：" +
      "，".join(f"{k} {v['prompts']} 个 prompt / {v['tokens']} token" for k, v in T["by_kind"].items()) + "。"
      "zh 全部来自本仓库设计文档（夹大量表格、代码、英文术语），en 来自模型卡与技术报告，code 是 C++ 与 Python。"
      "所以 kind 之间的差异同时混入了「领域/文体」差异，不是纯语言差异。\n")
    ck = P["total"].get("sim_equals_hitrate_sim_Lru")
    w(f"- 自检：栈距离 LRU 与 `hitrate_sim.Lru` 在主运行上逐步逐层结果{'完全相同' if ck else '不一致（需排查）'}。\n")

    # ---------------- part 1
    w("\n## 1. 逐层命中率（实测，主运行 " + P["name"] + f"，{P['total']['slots']} 槽）\n\n")
    w(f"整体 decode 命中 {pct(P['total']['decode_hit'])}，每 token {P['total']['decode_miss_per_tok']:.2f} 个 miss"
      f"（240 次访问）。LRU 复算 {pct(P['total']['sim_decode_hit'])}，{pct(P['total']['steps_exact'])} 的步所有 40 层命中数都与实测完全相同。\n")
    g_rows = []
    for name, rg in GROUPS:
        ls = [lay[l] for l in rg]
        dh = np.mean([x["decode_hit"] for x in ls])
        ph = np.nanmean([x["prefill_hit"] for x in ls])
        sh = sum(x["decode_miss_share"] for x in ls)
        mp = sum(x["decode_miss_per_tok"] for x in ls)
        g_rows.append([name, len(ls), pct(dh), pct(ph), f"{mp:.2f}", pct(sh)])
    w("\n按层组：\n\n" + md_table(["层组", "层数", "decode 命中", "prefill 命中", "miss/token（合计）", "占全部 decode miss"], g_rows))
    top5 = sorted(lay, key=lambda x: -x["decode_miss_share"])[:5]
    w("\n**最耗 NVMe 的 5 层**（decode miss 占比）：" +
      "，".join(f"L{x['layer']} {pct(x['decode_miss_share'])}（命中 {pct(x['decode_hit'])}）" for x in top5) +
      f"。均匀分布时每层应占 {pct(1 / L)}。\n")
    best5 = sorted(lay, key=lambda x: x["decode_miss_share"])[:5]
    w("命中最高的 5 层：" + "，".join(f"L{x['layer']} {pct(x['decode_hit'])}" for x in best5) + "。\n")
    conc = {r["layer"]: r for r in S["concentration"] if r["source"] == "trace_all"}
    concc = {r["layer"]: r for r in S["concentration"] if r["source"] == "chat_decode"}
    rows = []
    for x in lay:
        l = x["layer"]
        rows.append([l, pct(x["decode_hit"]), pct(x["prefill_hit"]), f"{x['decode_miss_per_tok']:.3f}",
                     pct(x["decode_miss_share"]), pct(x["sim_decode_hit"]), pct(x["step_agree"]),
                     conc[l]["distinct"], f"{conc[l]['eff_experts']:.0f}", conc[l]["n50"], conc[l]["n80"],
                     conc[l]["n95"], f"{concc[l]['eff_experts']:.0f}"])
    w("\n逐层明细（集中度列来自 traces/mixed 全部 27k token；最后一列是聊天 decode 步的有效专家数）：\n\n")
    w(md_table(["层", "decode 命中", "prefill 命中", "miss/token", "miss 占比", "LRU 复算", "逐步一致",
                "触及专家", "有效专家 2^H", "50% 质量", "80%", "95%", "聊天有效专家"], rows))
    # unusual layers
    effs = np.array([conc[l]["eff_experts"] for l in range(L)])
    w(f"\n集中度：有效专家数（熵的指数，上限 384）中位数 {np.median(effs):.0f}；"
      f"最平（接近均匀）的层：" + "，".join(f"L{l} {effs[l]:.0f}" for l in np.argsort(-effs)[:4]) +
      "；最集中的层：" + "，".join(f"L{l} {effs[l]:.0f}" for l in np.argsort(effs)[:4]) + "。")
    corr = float(np.corrcoef(effs, [x["decode_hit"] for x in lay])[0, 1])
    w(f"有效专家数与该层 decode 命中率的相关系数 {corr:+.2f}。\n")
    # layer x turn
    w("\n### 逐层 × 逐轮 decode 命中（主运行）\n\n完整矩阵见 `layer_turn_hits_<run>.csv`。这里给每个层组的均值：\n\n")
    lt = np.array(P["layer_turn"])
    hdr = ["层组"] + [f"{i + 1} {lab}" for i, lab in enumerate(P["labels"])]
    rows = [[name] + [pct(np.nanmean(lt[list(rg), i])) for i in range(lt.shape[1])] for name, rg in GROUPS]
    rows.append(["全部"] + [pct(np.nanmean(lt[:, i])) for i in range(lt.shape[1])])
    w(md_table(hdr, rows))
    w("\n轮次开头 vs 其余（实测，主运行）：\n\n")
    rows = [[s["turn"] + 1, s["label"], "是" if s["new_topic"] else "否", s["prefill_steps"],
             pct(s["first16_decode_hit"]), pct(s["first64_decode_hit"]), pct(s["rest_decode_hit"]),
             pct(s["first16_step_hit"])] for s in P["turn_starts"]]
    w(md_table(["轮", "标签", "新话题", "prefill 步", "前 16 decode", "前 64 decode", "其余 decode", "前 16 步(含 prefill)"], rows))
    for R in S["runs"][1:]:
        if R["total"]["gpu_prefill_min"] is not None and R["total"]["gpu_prefill_min"] < 512:
            w(f"\nGPU prefill 运行 {R['name']}：每个新话题轮次前 64 个 decode 步，实测 vs 纯 LRU 复算"
              "（差值 = GPU prefill 把本 prompt 的专家交接进缓存带来的命中，这就是「知道任务后预热」的可实现版本）：\n\n")
            w(md_table(["轮", "标签", "prefill", "实测 前64", "LRU 复算 前64", "实测 整轮", "LRU 复算 整轮"],
                       [[t["turn"] + 1, t["label"], t["prefill_mode"], pct(t["first64_decode_hit"]),
                         pct(t["sim_first64_decode_hit"]), pct(t["turn_decode_hit"]), pct(t["sim_turn_decode_hit"])]
                        for t in R["turn_starts"]]))
    w("\n验证：逐层「LRU 复算 vs 实测」一列在 5,000 和 5,400 两个容量上都应接近 100%；"
      "若某层偏差大，说明引擎在那一层做了 LRU 以外的事（预取、交接、pin）。\n")
    for R in S["runs"][1:]:
        dev = max(abs(a["sim_decode_hit"] - a["decode_hit"]) for a in R["layers"])
        w(f"- {R['name']}：LRU 复算 {pct(R['total']['sim_decode_hit'])} vs 实测 {pct(R['total']['decode_hit'])}，"
          f"逐步一致 {pct(R['total']['steps_exact'])}，单层最大偏差 {100 * dev:.2f} 个百分点。\n")

    # ---------------- part 2
    w("\n## 2. 任务分区\n\n")
    tt = tp["trace"]
    w("### 2.1 traces/mixed 的 code / en / zh（样本量大，结论可信度高）\n\n")
    rows = []
    for k in KINDS:
        f = tt["footprint"][k]
        rows.append([k, tt["ntok"][k], f["distinct_keys"], f["keys50"], f["keys80"], f["keys90"], f["keys95"]])
    w("每个任务覆盖其路由质量所需的 (层, 专家) 槽数（40 层合计，缓存容量 5,000–5,499）：\n\n")
    w(md_table(["任务", "token", "触及的键", "50%", "80%", "90%", "95%"], rows))
    w(f"\n三个任务 80% 集合的并集 {tt['unions']['union80']} 键、交集 {tt['unions']['intersection80']}；"
      f"90% 集合并集 {tt['unions']['union90']}、交集 {tt['unions']['intersection90']}。\n")
    rows = []
    for (a, b), v in tt["pair_avg"].items():
        rows.append([f"{a} vs {b}", f"{v['jaccard80']:.3f}", f"{v['weighted_overlap']:.3f}", f"{v['cosine']:.3f}", f"{v['jsd']:.3f}"])
    for n, v in tt["self_avg"].items():
        rows.append([f"{n} 自身两半（噪声基线）", f"{v['jaccard80']:.3f}", f"{v['weighted_overlap']:.3f}", f"{v['cosine']:.3f}", f"{v['jsd']:.3f}"])
    w("\n任务间重叠（40 层平均；80% 集合 Jaccard、加权重叠 Σmin(p,q)、频率向量余弦、JS 散度 bits）。"
      "「自身两半」是同一 kind 的 prompt 奇偶拆分，给出同任务内的自然差异：\n\n")
    w(md_table(["对", "Jaccard80", "加权重叠", "余弦", "JSD"], rows))
    sl = tt["spec_layer"]
    order = sorted(sl, key=lambda x: -x["jsd_excess"])
    w("\n**任务特异性最强的层**（任务间 JSD 减去同任务两半 JSD）：" +
      "，".join(f"L{x['layer']} {x['jsd_excess']:.3f}" for x in order[:5]) +
      "；**最任务无关的层**：" + "，".join(f"L{x['layer']} {x['jsd_excess']:.3f}" for x in order[-5:]) + "。\n")
    gr = []
    for name, rg in GROUPS:
        xs = [sl[l] for l in rg]
        gr.append([name, f"{np.mean([x['jsd_between'] for x in xs]):.3f}", f"{np.mean([x['jsd_within'] for x in xs]):.3f}",
                   f"{np.mean([x['jsd_excess'] for x in xs]):.3f}", f"{np.mean([x['core80_mass'] for x in xs]):.2f}",
                   sum(x["n_specific"] for x in xs)])
    w("\n" + md_table(["层组", "任务间 JSD", "同任务 JSD", "差值", "三任务共同 80% 核心占的质量", "特异专家数"], gr))
    ns = {k: sum(1 for r in tt["spec_rows"] if r["task"] == k) for k in KINDS}
    w(f"\n任务特异专家（份额比其他任务合计高 ≥3×，计数 ≥{S['params']['min_count_trace']}，且份额高于均匀 1/384）："
      + "，".join(f"{k} {ns[k]} 个" for k in KINDS) + "（完整列表 `task_specific_experts.csv`）。")
    rep = tt["replication"]
    w("奇偶 prompt 复现检验（偶数 prompt 上找到的特异专家，在奇数 prompt 上仍 ≥2× 的比例）：" +
      "，".join(f"{k} {rep[k]['still_2x_on_odd']}/{rep[k]['found_on_even']}（{pct(rep[k]['frac_2x'], 0)}）" for k in KINDS) + "。")
    sm = tt["spec_mass"]
    w("各任务落在「自己的特异专家」上的路由质量份额（全部层 / L0-1 / L2-19 / L20-39）：" +
      "，".join(f"{k} {pct(sm[k]['all'])} / " + " / ".join(pct(sm[k][g]) for g, _ in GROUPS) for k in KINDS) + "。\n")
    top_spec = sorted(tt["spec_rows"], key=lambda r: -r["count"] * min(r["ratio"], 50))[:12]
    if top_spec:
        w("\n支持度最高的特异专家示例：\n\n" + md_table(
            ["任务", "层", "专家", "次数", "本任务份额", "其他任务份额", "倍数"],
            [[r["task"], r["layer"], r["expert"], r["count"], pct(r["share_task"], 2), pct(r["share_others"], 2),
              "∞" if r["others_zero"] else f"{r['ratio']:.1f}"] for r in top_spec]))
    tc = tp["chat"]
    w("\n### 2.2 聊天 8 个轮次（每轮 85–490 步，样本小，噪声大）\n\n")
    rows = []
    for n in tc["ntok"]:
        f = tc["footprint"][n]
        rows.append([n, tc["ntok"][n], f["distinct_keys"], f["keys80"], f["keys90"]])
    w(md_table(["轮次", "步数（prefill+decode）", "触及的键", "80% 质量键数", "90%"], rows))
    w(f"\n聊天上按同一标准（计数 ≥{S['params']['min_count_chat']}、≥{S['params']['ratio']}×）找到的特异专家："
      + "，".join(f"{n} {sum(1 for r in tc['spec_rows'] if r['task'] == n)}" for n in tc["ntok"])
      + "。样本太小，其中相当一部分是噪声，只看趋势。\n")
    pa = tc["pair_avg"]
    vals = sorted(pa.items(), key=lambda kv: -kv[1]["weighted_overlap"])
    w("\n加权重叠最高的轮次对：" + "，".join(f"{a}/{b} {v['weighted_overlap']:.3f}" for (a, b), v in vals[:4]) +
      "；最低：" + "，".join(f"{a}/{b} {v['weighted_overlap']:.3f}" for (a, b), v in vals[-4:]) + "。")
    sa = tc["self_avg"]
    w("同一轮次前后两半的加权重叠（噪声基线）：" + "，".join(f"{n} {v['weighted_overlap']:.3f}" for n, v in sa.items()) + "。")
    kk = S["chat_kind_overlap"]
    w("\n聊天轮次与 trace 各 kind 的加权重叠（40 层平均），检验 trace 的 kind 画像能否迁移到真实对话：\n\n")
    w(md_table(["轮次"] + list(KINDS), [[n] + [f"{kk[n][k]:.3f}" for k in KINDS] for n in kk]))
    sl = tc["spec_layer"]
    order = sorted(sl, key=lambda x: -x["jsd_excess"])
    w("\n聊天数据上任务特异性最强的层：" + "，".join(f"L{x['layer']}" for x in order[:5]) +
      "；最弱：" + "，".join(f"L{x['layer']}" for x in order[-5:]) +
      f"。与 trace 排名的 Spearman 相关 {S['specificity_rank_corr']:+.2f}。\n")

    # ---------------- part 3
    w("\n## 3. 模拟（traces/mixed，精确全局 LRU）\n\n")
    caps = sim["caps"]
    show = [c for c in (3000, 4000, 5000, 5499, 6000, 7000) if c in caps]
    w("### 3.1 单任务流与混合顺序（冷启动，整段；括号内为去掉前 1,000 token 后的稳态）\n\n")
    rows = []
    for k in KINDS:
        s = sim["single"][k]
        rows.append([f"单任务 {k}（{s['tokens']} tok）"] + [f"{pct(s['hit'][c])} ({pct(s['hit_after1000'][c])})" for c in show])
    for n, s in sim["orders"].items():
        rows.append([n] + [f"{pct(s['hit'][c])} ({pct(s['hit_after1000'][c])})" for c in show])
    w(md_table(["流"] + [f"{c} 槽" for c in show], rows))
    ob = sim["orders"]
    c0 = 5499 if 5499 in caps else caps[-1]
    rr = ob["token_round_robin_3streams"]["hit_after1000"][c0]
    bl = ob["blocked_code_en_zh"]["hit_after1000"][c0]
    w(f"\n同样的 token，按块顺序（code→en→zh）稳态命中 {pct(bl)}，三路逐 token 轮转（相当于三个并发会话）{pct(rr)}；"
      f"差 {100 * (bl - rr):.2f} 个百分点 ≈ 每 token {(bl - rr) * 240:.2f} 个 miss ≈ "
      f"{tok_s((1 - bl) * 240):.2f} vs {tok_s((1 - rr) * 240):.2f} tok/s（近似）。\n")
    part = ob["token_round_robin_3streams"]["partitions"]
    w("\n三路轮转下「按任务静态分区」vs 共享 LRU（整段命中率）：\n\n")
    w(md_table(["容量", "共享 LRU", "三等分", "按工作集比例", "按 token 份额"],
               [[c, pct(part[c]["shared_lru"]), pct(part[c]["equal_thirds"]), pct(part[c]["by_working_set"]),
                 pct(part[c]["by_token_share"])] for c in show]))
    w("\n### 3.2 任务切换代价（A 的偶数 prompt 预热缓存 → 立即跑 B 的奇数 prompt，4 个起点平均）\n\n")
    for c in [x for x in (5000, 5499) if x in caps]:
        rows = []
        for a in list(KINDS) + ["cold"]:
            wn = "0-%d" % S["params"]["switch_n"]
            rows.append([a] + ["%s / %s" % (pct(sim["switch"][a + "->" + b][c]["0-64"]),
                                            pct(sim["switch"][a + "->" + b][c][wn])) for b in KINDS])
        w(f"**{c} 槽**，单元格 = B 的前 64 token / 前 {S['params']['switch_n']} token 命中率（行 A → 列 B；cold = 空缓存）：\n\n")
        w(md_table(["A \\ B"] + list(KINDS), rows) + "\n")
    w("对角线（A=B）是「没切换」的参照；同一列里对角线与非对角线之差就是切换代价。\n")
    w("\n### 3.3 任务感知策略\n\n")
    w("**(a) 切换时预热 B 的热门专家**（画像来自 B 的偶数 prompt，探测用奇数 prompt，无泄漏；"
      "'task' = B 自己的画像，'global' = 三个 kind 合并的画像，用来区分「任务感知」本身的价值）：\n\n")
    n = S["params"]["switch_n"]
    for c in [x for x in (5000, 5499) if x in caps]:
        rows = []
        for a in KINDS:
            for b in KINDS:
                if a == b:
                    continue
                base = sim["switch"][f"{a}->{b}"][c]
                cells = [f"{a}→{b}", pct(base["0-64"]), pct(base[f"0-{n}"])]
                for M in S["params"]["prewarm"]:
                    t_ = sim["prewarm"][f"{a}->{b}|task|{M}"][c]
                    g_ = sim["prewarm"][f"{a}->{b}|global|{M}"][c]
                    cells.append(f"{pct(t_[f'0-{n}'])} / {pct(g_[f'0-{n}'])} ({t_['prewarm_loads']:.0f})")
                rows.append(cells)
        w(f"{c} 槽（预热列 = 前 {n} token 命中 task / global，括号内为预热实际读入的专家数）：\n\n")
        w(md_table(["切换", "无预热 前64", f"无预热 前{n}"] + [f"预热 {M}" for M in S["params"]["prewarm"]], rows) + "\n")
    w("**(b) 静态 pin 任务 top-P 专家 + 其余 LRU**（单任务流，冷启动；'task' vs 'global' 画像；括号内为去掉前 1,000 token）：\n\n")
    rows = []
    for key, v in sim["pin"].items():
        for c in [x for x in (5000, 5499) if x in v]:
            rows.append([key.replace("|", " "), c, f"{pct(v[c]['lru_hit'])} ({pct(v[c]['lru_hit_after1000'])})",
                         f"{pct(v[c]['hit'])} ({pct(v[c]['hit_after1000'])})"])
    w(md_table(["任务 画像 P", "容量", "纯 LRU", "pin+LRU"], rows))
    w("\n**(c) 把预热搬到真实聊天运行上**（每个新话题轮次开始时，按标签的 kind 用 trace 画像预热 top-M；"
      "主运行路由，精确 LRU 重放）：\n\n")
    cp = S["chat_prewarm"]
    rows = []
    for c in cp["caps"]:
        b0 = cp["res"]["0"][str(c)]
        for M in cp["ms"]:
            r = cp["res"][str(M)][str(c)]
            rows.append([c, M, pct(b0["decode_hit"]), pct(r["decode_hit"]), pct(b0["newtopic_first64_decode_hit"]),
                         pct(r["newtopic_first64_decode_hit"]),
                         f"{tok_s(b0['decode_miss_per_tok']):.2f} → {tok_s(r['decode_miss_per_tok']):.2f}",
                         f"{r['prewarm_loads_total']}（{r['prewarm_loads_total'] * S['bytes_per_miss'] / 1e9:.0f} GB）"])
    w(md_table(["容量", "M", "decode 命中 基线", "预热后", "新话题前 64 基线", "预热后", "近似 tok/s", "预热读入专家（字节）"], rows))
    w("\n### 注意事项\n\n")
    for s in S["caveats"]:
        w(f"- {s}\n")
    w("\n## 文件\n\n")
    for f, d in S["files"].items():
        w(f"- `{f}`：{d}\n")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(lines))


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--trace", default="traces/mixed")
    ap.add_argument("--kinds", default="", help="JSON {kinds: {prompt_id: kind}} overriding state.pt")
    ap.add_argument("--out", default="bench/results/linux/expert_hits")
    ap.add_argument("--caps", default="3000,3500,4000,4500,5000,5499,6000,6500,7000")
    ap.add_argument("--switch-n", type=int, default=256)
    ap.add_argument("--prewarm", default="500,1000,2000,3000")
    ap.add_argument("--pin", default="1000,2000,3000")
    ap.add_argument("--min-count-trace", type=int, default=30)
    ap.add_argument("--min-count-chat", type=int, default=20)
    ap.add_argument("--ratio", type=float, default=3.0)
    a = ap.parse_args()
    os.chdir(REPO)
    os.makedirs(a.out, exist_ok=True)
    caps = sorted(set(int(c) for c in a.caps.split(",") if c))
    prewarm_ms = [int(x) for x in a.prewarm.split(",") if x]
    pin_ps = [int(x) for x in a.pin.split(",") if x]
    t_start = time.time()

    def log(msg):
        print(f"[{time.time() - t_start:6.1f}s] {msg}", flush=True)

    S = dict(generated=time.strftime("%Y-%m-%d %H:%M"), params=dict(
        caps=caps, switch_n=a.switch_n, prewarm=prewarm_ms, pin=pin_ps, min_count_trace=a.min_count_trace,
        min_count_chat=a.min_count_chat, ratio=a.ratio, ms_compute=MS_COMPUTE, ms_per_miss=MS_PER_MISS),
        layer_groups={n: [r.start, r.stop - 1] for n, r in GROUPS})
    files = {}

    # ---------------- part 1
    runs = []
    S["runs"] = []
    for i, d in enumerate(a.runs):
        run = load_run(d)
        rows, lt, starts, tot = analyse_run(run, caps, selfcheck=(i == 0))
        runs.append(run)
        log(f"run {run['name']}: {tot['records']} records, decode hit {tot['decode_hit']:.4f}, "
            f"sim {tot['sim_decode_hit']:.4f}, exact steps {tot['steps_exact']:.3f}"
            + (f", Lru check {tot['sim_equals_hitrate_sim_Lru']}" if i == 0 else ""))
        write_csv(os.path.join(a.out, f"layer_hits_{run['short']}.csv"), rows)
        lrows = [dict(layer=l, group=group_of(l), **{f"t{j + 1}_{lab.replace(' ', '_')}": lt[l, j]
                                                     for j, lab in enumerate(run["labels"])}) for l in range(L)]
        write_csv(os.path.join(a.out, f"layer_turn_hits_{run['short']}.csv"), lrows)
        write_csv(os.path.join(a.out, f"turn_starts_{run['short']}.csv"), starts)
        S["runs"].append(dict(name=run["name"], short=run["short"], labels=run["labels"], total=tot,
                              layers=rows, layer_turn=lt, turn_starts=starts))
    files["layer_hits_<run>.csv"] = "逐层实测 decode/prefill 命中、miss/token、miss 占比、LRU 复算（运行容量及 --caps 各容量）、逐步一致率"
    files["layer_turn_hits_<run>.csv"] = "40 层 × 8 轮 decode 命中矩阵"
    files["turn_starts_<run>.csv"] = "每轮前 16/32/64 步 vs 其余的命中"
    primary = runs[0]
    S["bytes_per_miss"] = S["runs"][0]["total"].get("bytes_per_miss") or 18.8e6

    # ---------------- trace
    pid, pos, tids = load_trace(a.trace)
    kinds, ksrc = prompt_kinds(a.trace, a.out, a.kinds)
    uniq = sorted(set(pid.tolist()))
    missing = [p for p in uniq if p not in kinds]
    if missing:
        raise SystemExit(f"prompts without a kind: {missing[:10]}")
    by_kind = {k: dict(prompts=sum(1 for p in uniq if kinds[p] == k),
                       tokens=int(sum((pid == p).sum() for p in uniq if kinds[p] == k))) for k in KINDS}
    S["trace"] = dict(dir=a.trace, tokens=len(pid), prompts=len(uniq), kind_source=ksrc, by_kind=by_kind)
    files["prompt_kinds.json"] = "trace 的 prompt→kind 映射（从 state.pt 提取）"
    log(f"trace: {len(pid)} tokens, {len(uniq)} prompts, kinds {ksrc}")

    # concentration
    conc_rows = []
    ctr = counts_of(tids)
    dec = primary["kind"] == "decode"
    cch = counts_of(primary["ids"][dec])
    for l in range(L):
        conc_rows.append(dict(source="trace_all", layer=l, group=group_of(l), tokens=len(tids), **conc_row(ctr[l])))
    for l in range(L):
        conc_rows.append(dict(source="chat_decode", layer=l, group=group_of(l), tokens=int(dec.sum()), **conc_row(cch[l])))
    write_csv(os.path.join(a.out, "layer_concentration.csv"), conc_rows)
    S["concentration"] = conc_rows
    files["layer_concentration.csv"] = "逐层路由集中度：触及专家数、熵、有效专家数、覆盖 50/80/95% 质量所需专家数、top1/top16 份额（trace 全部 & 聊天 decode）"

    # ---------------- part 2
    pid_ids = {p: tids[pid == p] for p in uniq}
    kind_prompts = {k: [p for p in dict.fromkeys(pid.tolist()) if kinds[p] == k] for k in KINDS}
    ttasks = {k: np.concatenate([pid_ids[p] for p in kind_prompts[k]]) for k in KINDS}
    thalves = {k: (np.concatenate([pid_ids[p] for p in kind_prompts[k][0::2]]),
                   np.concatenate([pid_ids[p] for p in kind_prompts[k][1::2]])) for k in KINDS}
    tt = task_tables("trace", ttasks, thalves, a.min_count_trace, a.ratio)
    tt["replication"] = replication(kind_prompts, pid_ids, a.min_count_trace, a.ratio)
    ctasks, chalves = {}, {}
    for i, lab in enumerate(primary["labels"]):
        m = (primary["turn"] == i) & (primary["kind"] != "startup")
        x = primary["ids"][m]
        if len(x):
            name = f"{i + 1}:{lab}"
            ctasks[name] = x
            h = len(x) // 2
            chalves[name] = (x[:h], x[h:])
    tc = task_tables("chat", ctasks, chalves, a.min_count_chat, a.ratio)
    log("task tables done")
    write_csv(os.path.join(a.out, "task_layer_stats.csv"), tt["stat_rows"] + tc["stat_rows"])
    write_csv(os.path.join(a.out, "task_overlap.csv"), tt["ov_rows"] + tc["ov_rows"])
    write_csv(os.path.join(a.out, "task_specific_experts.csv"), tt["spec_rows"] + tc["spec_rows"],
              ["source", "task", "layer", "expert", "count", "share_task", "share_others", "ratio", "others_zero"])
    write_csv(os.path.join(a.out, "layer_task_specificity.csv"), tt["spec_layer"] + tc["spec_layer"])
    files["task_layer_stats.csv"] = "任务 × 层：token 数、工作集（触及专家、每 128 token 窗口的平均触及数）、熵/有效专家、覆盖质量所需专家数、top-8 专家及份额"
    files["task_overlap.csv"] = "任务对 × 层：80% 集合 Jaccard、加权重叠、余弦、JSD；含 '#A/#B' 同任务两半的噪声基线"
    files["task_specific_experts.csv"] = f"任务特异专家（份额比 ≥{a.ratio}× 其他任务合计，带最小支持度）"
    files["layer_task_specificity.csv"] = "逐层任务特异性：任务间 JSD、同任务两半 JSD、差值、共同核心大小/质量、特异专家数"
    # chat turns vs trace kinds
    tk = {k: counts_of(ttasks[k]) for k in KINDS}
    S["chat_kind_overlap"] = {n: {k: float(np.mean([wov(norm(tc["cnt"][n][l]), norm(tk[k][l])) for l in range(L)]))
                                  for k in KINDS} for n in ctasks}
    ra = np.argsort(np.argsort([x["jsd_excess"] for x in tt["spec_layer"]]))
    rb = np.argsort(np.argsort([x["jsd_excess"] for x in tc["spec_layer"]]))
    S["specificity_rank_corr"] = float(np.corrcoef(ra, rb)[0, 1])

    def strip(t):
        return dict(ntok=t["ntok"], pair_avg={f"{x}|{y}": v for (x, y), v in t["pair_avg"].items()},
                    self_avg=t["self_avg"], spec_layer=t["spec_layer"], footprint=t["footprint"],
                    unions=t["unions"], spec_mass=t["spec_mass"], n_specific={n: sum(1 for r in t["spec_rows"] if r["task"] == n) for n in t["ntok"]},
                    overlap_by_layer={f"{r['a']}|{r['b']}": [] for r in t["ov_rows"]})
    tasks_json = {}
    for src, t in (("trace", tt), ("chat", tc)):
        j = strip(t)
        for r in t["ov_rows"]:
            j["overlap_by_layer"][f"{r['a']}|{r['b']}"].append(
                dict(jaccard80=r["jaccard80"], weighted_overlap=r["weighted_overlap"], cosine=r["cosine"], jsd=r["jsd"]))
        j["top_experts"] = {n: [[int(e) for e in np.argsort(-t["cnt"][n][l], kind="stable")[:8]] for l in range(L)]
                            for n in t["ntok"]}
        if src == "trace":
            j["replication"] = t["replication"]
        tasks_json[src] = j
    S["tasks_json"] = tasks_json
    report_tasks = {"trace": tt, "chat": tc}

    # ---------------- part 3
    log("simulating the trace ...")
    sim, kp = simulate_trace(pid, tids, kinds, caps, a.switch_n, prewarm_ms, pin_ps, log)
    sim_rows = []
    for k, s in sim["single"].items():
        for c in caps:
            sim_rows.append(dict(scenario=f"single_{k}", cap=c, hit=s["hit"][c], hit_after1000=s["hit_after1000"][c],
                                 miss_per_tok=s["miss_per_tok"][c], approx_tok_s=tok_s(s["miss_after1000"][c])))
    for n, s in sim["orders"].items():
        for c in caps:
            sim_rows.append(dict(scenario=n, cap=c, hit=s["hit"][c], hit_after1000=s["hit_after1000"][c],
                                 miss_per_tok=s["miss_per_tok"][c], approx_tok_s=tok_s(s["miss_after1000"][c])))
    write_csv(os.path.join(a.out, "sim_curves.csv"), sim_rows)
    sw_rows = []
    for key, v in sim["switch"].items():
        a_, b_ = key.split("->")
        for c in caps:
            sw_rows.append(dict(policy="lru", prewarm_m=0, profile="", a=a_, b=b_, cap=c, **v[c]))
    for key, v in sim["prewarm"].items():
        ab, prof, M = key.split("|")
        a_, b_ = ab.split("->")
        for c in caps:
            sw_rows.append(dict(policy="prewarm", prewarm_m=int(M), profile=prof, a=a_, b=b_, cap=c, **v[c]))
    write_csv(os.path.join(a.out, "sim_switch.csv"), sw_rows,
              ["policy", "prewarm_m", "profile", "a", "b", "cap"] + sim["switch_windows"] + ["prewarm_loads"])
    pin_rows = []
    for key, v in sim["pin"].items():
        b_, prof, P = key.split("|")
        for c, r in v.items():
            pin_rows.append(dict(task=b_, profile=prof, pinned=int(P), cap=c, **r))
    part_rows = [dict(cap=c, **v) for c, v in sim["orders"]["token_round_robin_3streams"]["partitions"].items()]
    write_csv(os.path.join(a.out, "sim_pin.csv"), pin_rows)
    write_csv(os.path.join(a.out, "sim_partition.csv"), part_rows)
    files["sim_curves.csv"] = "单任务流与四种混合顺序在各容量下的命中率（整段/去掉前 1000 token）、miss/token、近似 tok/s"
    files["sim_switch.csv"] = "切换矩阵 A→B 与预热策略：窗口命中率（0-16/0-64/0-N/N-2N）和预热读入数"
    files["sim_pin.csv"] = "pin 任务 top-P + LRU vs 纯 LRU"
    files["sim_partition.csv"] = "三路轮转下的静态分区 vs 共享 LRU"

    log("chat pre-warm replay ...")
    ccaps = sorted(set([primary["slots"], 5000, 5499]))
    cms = [500, 1000, 2000]
    cp = chat_prewarm(primary, tk, cms, ccaps)
    S["chat_prewarm"] = dict(run=primary["name"], caps=ccaps, ms=cms,
                             res={str(M): {str(c): v for c, v in r.items()} for M, r in cp.items()})
    write_csv(os.path.join(a.out, "sim_chat_prewarm.csv"),
              [dict(prewarm_m=M, cap=c, **v) for M, r in cp.items() for c, v in r.items()])
    files["sim_chat_prewarm.csv"] = "在主聊天运行上重放：新话题轮次开始时按 kind 画像预热 top-M"

    # ---------------- conclusions (data-driven)
    c0 = 5499 if 5499 in caps else caps[len(caps) // 2]
    P0 = S["runs"][0]
    n = a.switch_n
    concl = []
    t0_ = P0["total"]
    top5 = sorted(P0["layers"], key=lambda x: -x["decode_miss_share"])[:5]
    g01 = sum(x["decode_miss_share"] for x in P0["layers"][:2])
    concl.append(f"逐层（实测，{P0['name']}，{t0_['slots']} 槽）：decode 命中 {pct(t0_['decode_hit'])}，"
                 f"{t0_['decode_miss_per_tok']:.2f} miss/token；LRU 复算逐步逐层 {pct(t0_['steps_exact'])} 一致。"
                 "最耗 NVMe 的 5 层 " + "、".join(f"L{x['layer']}({pct(x['decode_miss_share'])})" for x in top5) +
                 f"，合计 {pct(sum(x['decode_miss_share'] for x in top5))}；L0-1 两层占 {pct(g01)}（均匀应为 5%）。"
                 + ("miss 在 40 层上分布相当平（最贵一层也不到均匀份额的 2 倍），没有单层能「修一层救全局」。"
                  if top5[0]["decode_miss_share"] < 2.0 / L else "miss 明显集中在少数层。"))
    effs = [r["eff_experts"] for r in conc_rows if r["source"] == "trace_all"]
    corr = float(np.corrcoef(effs, [x["decode_hit"] for x in P0["layers"]])[0, 1])
    concl.append(f"集中度：每层有效专家数（2^熵，上限 384）从 {min(effs):.0f}（L{int(np.argmin(effs))}）到 {max(effs):.0f}"
                 f"（L{int(np.argmax(effs))}），与该层命中率相关 {corr:+.2f}；"
                 "最平的层 " + "、".join(f"L{l}" for l in np.argsort(effs)[::-1][:5]) +
                 "，最集中的层 " + "、".join(f"L{l}" for l in np.argsort(effs)[:5]) + "。")
    sl = sorted(tt["spec_layer"], key=lambda x: -x["jsd_excess"])
    pv = tt["pair_avg"]
    wo = [v["weighted_overlap"] for v in pv.values()]
    wself = [v["weighted_overlap"] for v in tt["self_avg"].values()]
    concl.append(f"任务分区（trace）：kind 之间逐层加权重叠 {min(wo):.2f}–{max(wo):.2f}，同 kind 两半 {min(wself):.2f}–{max(wself):.2f}；"
                 f"三个 kind 的 80% 集合交集 {tt['unions']['intersection80']} 键、并集 {tt['unions']['union80']} 键。"
                 "最任务特异的层 " + "、".join(f"L{x['layer']}" for x in sl[:5]) +
                 "，最任务无关的层 " + "、".join(f"L{x['layer']}" for x in sl[-5:]) +
                 "。各 kind 份额最大的专属专家：" +
                 "，".join((lambda r: f"{k} L{r['layer']}/e{r['expert']} 占 {pct(r['share_task'])}（其他 {pct(r['share_others'], 2)}）")(
                     max((r for r in tt["spec_rows"] if r["task"] == k), key=lambda r: r["share_task"]))
                     for k in KINDS if any(r["task"] == k for r in tt["spec_rows"])) + "。")
    # switch cost
    offd, diag64, diagN = [], [], []
    for b in KINDS:
        dg = sim["switch"][f"{b}->{b}"][c0]
        for a_ in KINDS:
            if a_ != b:
                o = sim["switch"][f"{a_}->{b}"][c0]
                offd.append((dg["0-64"] - o["0-64"], dg[f"0-{n}"] - o[f"0-{n}"]))
    m64 = float(np.mean([x[0] for x in offd]))
    mN = float(np.mean([x[1] for x in offd]))
    cold = float(np.mean([sim["switch"][f"{b}->{b}"][c0]["0-64"] - sim["switch"][f"cold->{b}"][c0]["0-64"] for b in KINDS]))
    concl.append(f"切换代价（{c0} 槽）：换 kind 后前 64 token 命中比不换低 {100 * m64:.1f} 个百分点，前 {n} token 平均低 {100 * mN:.1f} 个百分点"
                 f"（≈ 每次切换多 {mN * 240 * n:.0f} 个 miss ≈ {mN * 240 * n * MS_PER_MISS / 1000:.1f} s）；"
                 f"对比空缓存冷启动前 64 token 低 {100 * cold:.1f} 个百分点——跨任务共享的核心专家让大部分缓存在切换后仍然有用。")
    best = None
    for key, v in sim["prewarm"].items():
        ab, prof_, M = key.split("|")
        a_, b_ = ab.split("->")
        if a_ == b_ or a_ == "cold":
            continue
        g = v[c0][f"0-{n}"] - sim["switch"][ab][c0][f"0-{n}"]
        saved = g * 240 * n
        loads = v[c0]["prewarm_loads"]
        if best is None or g > best[0]:
            best = (g, key, loads, saved)
    tg = np.mean([sim["prewarm"][k][c0][f"0-{n}"] - sim["prewarm"][k.replace("|task|", "|global|")][c0][f"0-{n}"]
                  for k in sim["prewarm"] if "|task|" in k and k.split("->")[0] != "cold"
                  and k.split("->")[0] != k.split("->")[1].split("|")[0]])
    if best:
        g, key, loads, saved = best
        concl.append(f"切换时预热（{c0} 槽）：最好的 {key} 让前 {n} token 命中 +{100 * g:.2f} 个百分点，省 ≈{saved:.0f} 个 miss，"
                     f"却要先读入 {loads:.0f} 个专家（≈{loads * S['bytes_per_miss'] / 1e9:.0f} GB）——读入数 > 省下的 miss，"
                     f"只有完全放在空闲时间才不亏。任务画像比全局画像平均只多 {100 * tg:+.2f} 个百分点：收益几乎全来自「热门专家」本身，不是「任务感知」。")
    cp0 = cp[0][primary["slots"]]
    cpb = max((cp[M][primary["slots"]] for M in cms), key=lambda r: r["decode_hit"])
    concl.append(f"把 trace 画像的预热搬到真实聊天上（{primary['slots']} 槽）：decode 命中 {pct(cp0['decode_hit'])} → 最好 {pct(cpb['decode_hit'])}"
                 f"（近似 {tok_s(cp0['decode_miss_per_tok']):.2f} → {tok_s(cpb['decode_miss_per_tok']):.2f} tok/s），没有收益——"
                 "聊天轮次与 trace 同 kind 的加权重叠只有 "
                 + f"{min(v[label_kind(n.split(':', 1)[1])] for n, v in S['chat_kind_overlap'].items()):.2f}–"
                 + f"{max(v[label_kind(n.split(':', 1)[1])] for n, v in S['chat_kind_overlap'].items()):.2f}"
                 + "（同轮次两半 " + f"{min(v['weighted_overlap'] for v in tc['self_avg'].values()):.2f}–"
                 + f"{max(v['weighted_overlap'] for v in tc['self_avg'].values()):.2f}），文档画像迁移不到生成的回答上。")
    pm = [R for R in S["runs"] if R["total"]["gpu_prefill_min"] is not None and R["total"]["gpu_prefill_min"] < 512]
    if pm:
        R = pm[0]
        concl.append(f"可实现的「任务感知预热」其实已经存在：GPU prefill 把本 prompt 的专家交接进缓存。{R['name']} 上实测 decode 命中 "
                     f"{pct(R['total']['decode_hit'])}，比纯 LRU 复算 {pct(R['total']['sim_decode_hit'])} 高 "
                     f"{100 * (R['total']['decode_hit'] - R['total']['sim_decode_hit']):.1f} 个百分点。")
    part = sim["orders"]["token_round_robin_3streams"]["partitions"][c0]
    ob = sim["orders"]
    pins = {k: 100 * (v[c0]["hit_after1000"] - v[c0]["lru_hit_after1000"]) for k, v in sim["pin"].items() if "|task|" in k and c0 in v}
    concl.append(f"静态分区大幅负收益：三路逐 token 轮转、{c0} 槽，共享 LRU {pct(part['shared_lru'])}，三等分 {pct(part['equal_thirds'])}、"
                 f"按工作集 {pct(part['by_working_set'])}（共享核心专家在每个分区里各占一份，每个分区又都小于单任务工作集）。"
                 f"pin 任务 top-P（P={','.join(str(x) for x in pin_ps)}）+ LRU 的稳态变化在 {min(pins.values()):+.1f} 到 {max(pins.values()):+.1f} 个百分点之间（"
                 + "、".join(f"{k.split('|')[0]} P={k.split('|')[2]} {x:+.1f}" for k, x in pins.items()) + "）；"
                 + (lambda k: (lambda v: f"最好的 {k} 近似 {tok_s((1 - v['lru_hit_after1000']) * 240):.2f} → {tok_s((1 - v['hit_after1000']) * 240):.2f} tok/s")(sim["pin"][k][c0]))(max(pins, key=pins.get))
                 + "，而且只在一个 kind 上为正、其余为负，还要事先知道任务——收益不稳定，不值得做。")
    concl.append(f"真正的代价在交错：同样的 27k token，按块顺序 {pct(ob['blocked_code_en_zh']['hit_after1000'][c0])}，"
                 f"三路逐 token 轮转（三个并发会话）{pct(ob['token_round_robin_3streams']['hit_after1000'][c0])}，"
                 f"近似 {tok_s(ob['blocked_code_en_zh']['miss_after1000'][c0]):.2f} → {tok_s(ob['token_round_robin_3streams']['miss_after1000'][c0]):.2f} tok/s（单流口径）。"
                 "按 prompt 轮转（一次跑完一个请求）与块顺序几乎一样。")
    S["conclusions"] = concl
    S["caveats"] = [
        "聊天数据只有一条路由流（四个 auto_ab 运行逐 token 相同），8 个轮次每轮 85–490 步；每层每轮只有 ~500–2,900 次路由，"
        "逐专家的份额噪声很大，所以任务特异专家与重叠度的主要结论以 traces/mixed 为准，聊天部分看同任务两半的噪声基线再下结论。",
        "traces/mixed 是 teacher-forced 的文档文本（非模型自己生成的回答），而且 zh 全是本仓库设计文档、en 是模型卡/技术报告、"
        "code 是 C++/Python：kind 的差异是「语言 + 领域 + 文体」的混合，真实用户的中文闲聊/写作与这里的 zh 差很多。",
        "模拟只建模 demand LRU：没有预取、engram、backfill、prefill 交接；切换实验里 A 的预热用 A 的全部偶数 prompt（~3–5k token，"
        "远超缓存容量，已是稳态）。",
        "任务感知策略（预热/pin/分区）需要在请求到达时知道任务类型（例如按语言/是否代码做分类器，或由客户端声明），"
        "以及每个任务的离线路由画像；预热读入的字节会与 P0 demand 读争带宽，必须放在空闲时段或 P3 优先级。",
        f"tok/s 换算用 ms/token ≈ {MS_COMPUTE:.0f} + {MS_PER_MISS} × miss/token，忽略了 miss 在层间的重叠与 IO 并发，是近似值；"
        "按 CLAUDE.md，任何预测收益先减半。",
        "pfmin/4_newdefault 的 GPU prefill 会把专家交接进缓存，而 route.bin 不记录这些插入，所以它的 LRU 复算与实测不一致是预期的。",
    ]
    S["files"] = files

    # ---------------- write
    S["trace"]["by_kind"] = by_kind
    S["sim"] = sim
    summary = dict(generated=S["generated"], params=S["params"], layer_groups=S["layer_groups"],
                   runs=[dict(name=R["name"], labels=R["labels"], total=R["total"], layers=R["layers"],
                              layer_turn=R["layer_turn"], turn_starts=R["turn_starts"]) for R in S["runs"]],
                   concentration=S["concentration"], trace=S["trace"], tasks=S["tasks_json"],
                   chat_kind_overlap=S["chat_kind_overlap"], specificity_rank_corr=S["specificity_rank_corr"],
                   sim={k: v for k, v in sim.items() if k != "halves"}, sim_halves=sim["halves"],
                   chat_prewarm=S["chat_prewarm"], bytes_per_miss=S["bytes_per_miss"],
                   conclusions=S["conclusions"], caveats=S["caveats"], files=files)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(r4(summary), f, ensure_ascii=False, indent=1, allow_nan=False, default=str)
    S["tasks"] = report_tasks
    for k in ("trace", "chat"):
        report_tasks[k]["pair_avg"] = report_tasks[k]["pair_avg"]
    write_report(os.path.join(a.out, "report.md"), S)
    log(f"wrote {a.out}/report.md, summary.json and {len(files)} CSV groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
