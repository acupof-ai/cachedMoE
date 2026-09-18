#!/usr/bin/env python3
"""Resident-only routing: what does decode lose if it never waits for an expert?

Track Y (P4). Context: decode on this machine is NVMe-bound. Each routed expert is
18.8 MB and costs ~4.18 ms off the drive; the resident compute is ~2 ms/layer, so a
40-layer token is ~80 ms of GPU and ~110 ms of stall at LRU's 0.89 hit rate.
Speculative decoding does not help, because verifying a block of drafts fetches the
union of their experts (docs/p4_dspark_runtime.md 6.5): the miss stream is unchanged.

The idea simulated here is different, and lossy: **route only into experts that are
already resident**. V4.1-Flash's gate picks 6 of 384 per layer and normalises the six
raw scores (inference/model.py Gate, norm_topk_prob = true). If some of the six are
not in the cache, drop them and renormalise the gate weights over the resident ones.
The drive keeps backfilling in the background, so no step ever stalls.

The question this tool answers is *how much gate-weight mass that throws away*,
because that -- not the tok/s, which is trivially 1000/80 = 12.5 -- is what decides
whether the output is still the model's.

Three variants:

  exact          the baseline: every miss is fetched and the GPU waits (what
                 tools/cache_sim.py calls the demand path). Establishes the residency
                 statistics and the stall.
  verify-only    DSpark block of 5: position 1 of each block routes exactly (fetches,
                 warms the cache), positions 2..5 route resident-only. The block costs
                 one stalled step plus one free step and yields ~4.7 tokens.
  backfill       every step is resident-only. Misses are queued and served at the
                 drive's own rate (4.5 GB/s / 18.8 MB = 239 experts/s, i.e. ~19 per
                 80 ms step), so the cache still refreshes -- but LRU now only sees
                 the experts that were actually *served*, which is modelled here.

Usage:
    uv run python tools/resident_sim.py --trace traces/mixed
        --capacities 4500,5711,8000 --out bench/results/resident
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import OrderedDict, deque

import numpy as np

EXPERT_BYTES = 18_800_640
N_ROUTED_PER_LAYER = 384
TOPK = 6                       # config.json text_config.num_experts_per_tok
DSPARK_BLOCK = 5               # config.json text_config.dspark_block_size


# --------------------------------------------------------------------------- #
# trace
# --------------------------------------------------------------------------- #

def load_trace(specs, max_tokens=0, quiet=False):
    """(layer, prompt, pos, top6 ids, top6 raw gate scores) in decode order.

    Only the five columns this tool needs are read; `tools/route_trace.py` guarantees
    top16_ids[:6] == top6_ids (ties included), so top16_scores[:6] is exactly the six
    raw scores the gate would normalise.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = []
    for spec in specs:
        if os.path.isdir(spec):
            files += sorted(glob.glob(os.path.join(spec, "route_layer*.parquet")))
        elif any(c in spec for c in "*?["):
            files += sorted(glob.glob(spec))
        else:
            files.append(spec)
    if not files:
        raise SystemExit("no trace files matched %s" % (specs,))

    cols = ["layer", "prompt_id", "pos", "top6_ids", "top16_scores"]
    table = pa.concat_tables([pq.read_table(f, columns=cols) for f in files],
                             promote_options="default")

    def fsl(name, dtype):
        arr = table.column(name).combine_chunks()
        w = arr.type.list_size
        return arr.flatten().to_numpy(zero_copy_only=False).astype(dtype).reshape(-1, w)

    layer = table.column("layer").to_numpy(zero_copy_only=False).astype(np.int32)
    prompt = table.column("prompt_id").to_numpy(zero_copy_only=False).astype(np.int32)
    pos = table.column("pos").to_numpy(zero_copy_only=False).astype(np.int64)
    order = np.lexsort((layer, pos, prompt))

    top6 = fsl("top6_ids", np.int32)[order]
    scores = fsl("top16_scores", np.float32)[order][:, :TOPK]
    layer, prompt, pos = layer[order], prompt[order], pos[order]
    n_layers = int(layer.max()) + 1

    if max_tokens:
        keep = max_tokens * n_layers
        layer, prompt, pos = layer[:keep], prompt[:keep], pos[:keep]
        top6, scores = top6[:keep], scores[:keep]

    keys = layer[:, None] * N_ROUTED_PER_LAYER + top6
    if not quiet:
        print("trace: %d rows, %d tokens x %d layers, top-%d"
              % (len(layer), len(layer) // n_layers, n_layers, scores.shape[1]))
    return dict(layer=layer, prompt=prompt, pos=pos, keys=keys, scores=scores,
                top6=top6, n_layers=n_layers)


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #

class Stats:
    """Residency and lost-weight accumulators, split by how the step routed."""

    def __init__(self, n_layers):
        self.n_layers = n_layers
        self.hist = np.zeros((n_layers, TOPK + 1), dtype=np.int64)   # resident count
        self.mass_lost = np.zeros(n_layers)      # sum over rows of lost score fraction
        self.mass_lost_sq = np.zeros(n_layers)
        self.top1_lost = np.zeros(n_layers, dtype=np.int64)
        self.rows = np.zeros(n_layers, dtype=np.int64)
        self.served = 0
        self.requested = 0
        self.tok_all_resident = 0                # tokens with 6/6 at *every* layer
        self.tokens = 0

    def add(self, L, n_res, lost, top1_missing):
        self.hist[L, n_res] += 1
        self.mass_lost[L] += lost
        self.mass_lost_sq[L] += lost * lost
        self.top1_lost[L] += top1_missing
        self.rows[L] += 1
        self.served += n_res
        self.requested += TOPK

    def report(self, label):
        rows = int(self.rows.sum())
        if rows == 0:
            return {"label": label, "rows": 0}
        h = self.hist.sum(axis=0)
        denom = np.maximum(self.rows, 1)
        per_layer_lost = np.where(self.rows > 0, self.mass_lost / denom, 0.0)
        per_layer_full = np.where(self.rows > 0, self.hist[:, TOPK] / denom, 0.0)
        worst = np.argsort(-per_layer_lost)[:6]
        groups = {}
        for lo in range(0, self.n_layers, 10):
            hi = min(lo + 10, self.n_layers)
            gh = self.hist[lo:hi].sum(axis=0)
            gr = max(1, int(self.rows[lo:hi].sum()))
            groups["L%02d-%02d" % (lo, hi - 1)] = {
                "resident_count_frac": [round(float(x) / gr, 4) for x in gh],
                "mean_resident": round(float((gh * np.arange(TOPK + 1)).sum()) / gr, 3),
                "mass_lost": round(float(self.mass_lost[lo:hi].sum()) / gr, 5),
            }
        mean = float(self.mass_lost.sum()) / rows
        var = max(0.0, float(self.mass_lost_sq.sum()) / rows - mean * mean)
        return {
            "label": label,
            "rows": rows,
            "resident_count_frac": [round(float(x) / rows, 4) for x in h],
            "mean_resident_of_k": round(float((h * np.arange(TOPK + 1)).sum()) / rows, 3),
            "frac_rows_all_k_resident": round(float(h[TOPK]) / rows, 4),
            "frac_tokens_all_k_all_layers": round(self.tok_all_resident / max(1, self.tokens), 4),
            "served_frac": round(self.served / max(1, self.requested), 4),
            "mean_mass_lost": round(mean, 5),
            "sd_mass_lost": round(var ** 0.5, 5),
            "frac_rows_top1_missing": round(float(self.top1_lost.sum()) / rows, 5),
            "layer_groups": groups,
            "mass_lost_per_layer": [round(float(x), 5) for x in per_layer_lost],
            "frac_all_k_per_layer": [round(float(x), 4) for x in per_layer_full],
            "worst_layers": [{"layer": int(l),
                              "mass_lost": round(float(per_layer_lost[l]), 5),
                              "frac_all_k": round(float(per_layer_full[l]), 4)}
                             for l in worst],
        }


def replay(tr, capacity, mode, args):
    """One pass of the trace through an LRU of `capacity` slots.

    mode = "exact"        every miss is fetched, the GPU waits (baseline)
           "verify-only"  pos % 5 == 0 routes exactly, the other four resident-only
           "backfill"     every row resident-only; misses go to a background queue
                          drained at `--backfill-per-token` experts per 80 ms step

    LRU is touched only by experts that were actually *used* (resident, or admitted by
    a backfill that landed), which is the whole point: in resident-only mode the policy
    never learns about the experts it skipped, so the residency it sustains is not the
    residency the exact replay reports.
    """
    layer, pos, keys, scores = tr["layer"], tr["pos"], tr["keys"], tr["scores"]
    n_layers = tr["n_layers"]
    rows = len(layer)
    warm_rows = int(rows * args.warmup_frac)

    od = OrderedDict()
    exact_st, lossy_st = Stats(n_layers), Stats(n_layers)
    queue, queued = deque(), set()
    backfill_credit = 0.0
    per_row = args.backfill_per_token / n_layers      # experts admitted per row
    stall_ms = 0.0
    exact_steps = lossy_steps = 0
    fetched_demand = fetched_backfill = 0
    queue_drops = 0
    tok_ok_exact = tok_ok_lossy = True
    lossy_now = False

    for i in range(rows):
        L = int(layer[i])
        counted = i >= warm_rows
        if mode == "exact":
            lossy = False
        elif mode == "verify-only":
            lossy = (int(pos[i]) % DSPARK_BLOCK) != 0
        else:
            lossy = True

        if L == 0:
            lossy_now = lossy
            tok_ok_exact = tok_ok_lossy = True
            if counted:
                if lossy:
                    lossy_st.tokens += 1
                    lossy_steps += 1
                else:
                    exact_st.tokens += 1
                    exact_steps += 1
        lossy = lossy_now if mode != "exact" else False

        # --- background backfill: the drive keeps working while the GPU does not wait
        if mode != "exact":
            backfill_credit += per_row
            while backfill_credit >= 1.0 and queue:
                k = queue.popleft()
                queued.discard(k)
                backfill_credit -= 1.0
                if counted:
                    fetched_backfill += 1
                if k not in od:
                    od[k] = None
                    if len(od) > capacity:
                        od.popitem(last=False)

        row_keys = keys[i]
        row_scores = scores[i]
        total = float(row_scores.sum()) or 1.0
        n_res = 0
        lost = 0.0
        top1_missing = 0
        misses = []
        for j in range(TOPK):
            k = int(row_keys[j])
            if k in od:
                od.move_to_end(k)
                n_res += 1
            else:
                misses.append(k)
                lost += float(row_scores[j])
                if j == 0:
                    top1_missing = 1
        lost /= total

        if not lossy:
            # exact routing: fetch the misses, the GPU stalls, everything is admitted
            if misses:
                m = len(misses)
                service = max(args.t_io_ms, m / args.io_qd * args.t_io_ms,
                              m * EXPERT_BYTES / (args.nvme_gbps * 1e9) * 1e3)
                if counted:
                    stall_ms += service
                    fetched_demand += m
                for k in misses:
                    od[k] = None
                    od.move_to_end(k)
                    if len(od) > capacity:
                        od.popitem(last=False)
            if counted:
                exact_st.add(L, n_res, lost, top1_missing)
                if n_res < TOPK:
                    tok_ok_exact = False
                if L == n_layers - 1 and tok_ok_exact:
                    exact_st.tok_all_resident += 1
        else:
            # resident-only: skip the misses, queue them for the drive
            for k in misses:
                if k in queued:
                    continue
                if len(queue) >= args.queue_cap:
                    old = queue.popleft()
                    queued.discard(old)
                    queue_drops += 1
                queue.append(k)
                queued.add(k)
            if counted:
                lossy_st.add(L, n_res, lost, top1_missing)
                if n_res < TOPK:
                    tok_ok_lossy = False
                if L == n_layers - 1 and tok_ok_lossy:
                    lossy_st.tok_all_resident += 1

    tokens_counted = max(1, (rows - warm_rows) // n_layers)
    t_compute = args.t_layer_ms * n_layers
    out = {
        "mode": mode,
        "capacity": capacity,
        "tokens": tokens_counted,
        "exact_steps": exact_steps,
        "resident_only_steps": lossy_steps,
        "stall_ms_total": round(stall_ms, 1),
        "experts_fetched_demand": fetched_demand,
        "experts_fetched_backfill": fetched_backfill,
        "queue_drops": queue_drops,
        "queue_len_end": len(queue),
        "t_compute_ms": t_compute,
    }
    if exact_st.rows.sum():
        out["exact"] = exact_st.report("exact-routed steps")
        out["stall_ms_per_exact_step"] = round(stall_ms / max(1, exact_steps), 1)
    if lossy_st.rows.sum():
        out["resident_only"] = lossy_st.report("resident-only steps")

    # --- throughput -------------------------------------------------------
    if mode == "exact":
        ms = t_compute + stall_ms / tokens_counted
        out["ms_per_token"] = round(ms, 1)
        out["tokens_per_s"] = round(1000.0 / ms, 2)
    elif mode == "verify-only":
        # one stalled exact step + one free verify step per block, ~3.7 accepted + 1
        block_ms = (t_compute + out["stall_ms_per_exact_step"]) + t_compute
        yld = 1.0 + args.accepted_per_block
        out["block_ms"] = round(block_ms, 1)
        out["tokens_per_block"] = yld
        out["ms_per_token"] = round(block_ms / yld, 1)
        out["tokens_per_s"] = round(1000.0 * yld / block_ms, 2)
    else:
        out["ms_per_token"] = t_compute
        out["tokens_per_s"] = round(1000.0 / t_compute, 2)
    return out


# --------------------------------------------------------------------------- #
# router groups (question 4)
# --------------------------------------------------------------------------- #

def group_analysis(tr, model_config, n_groups_probe=(8, 16, 32, 64)):
    """V4.1-Flash has no n_group / topk_group -- confirm it, then show why it matters.

    config.json carries topk_method = "noaux_tc" but *no* n_group / topk_group keys,
    and inference/model.py Gate does a flat `(scores + bias).topk(self.topk)` over all
    384 experts with no group mask. So there is no routing group to cache or evict at.
    What is measured instead is the only thing a group-granular cache could exploit: if
    the 384 ids were cut into G contiguous blocks, how many distinct blocks does one
    token-layer's six experts touch? Near-six means a block is never a useful fetch
    unit, and the amplification column prices it.
    """
    cfg = {}
    tc = (model_config or {}).get("text_config", {})
    for k in ("n_group", "topk_group", "topk_method", "scoring_func",
              "num_experts_per_tok", "n_routed_experts", "norm_topk_prob",
              "routed_scaling_factor", "n_shared_experts"):
        cfg[k] = tc.get(k, None)
    out = {"config": cfg,
           "group_limited_routing": bool(tc.get("n_group") or tc.get("topk_group"))}
    ids = tr["top6"]
    sample = ids[: min(len(ids), 200_000)]
    occ = {}
    for g in n_groups_probe:
        width = N_ROUTED_PER_LAYER // g
        blk = sample // width
        blk_sorted = np.sort(blk, axis=1)
        distinct = 1 + (np.diff(blk_sorted, axis=1) != 0).sum(axis=1)
        occ["G=%d" % g] = {
            "experts_per_block": width,
            "mean_distinct_blocks_per_row": round(float(distinct.mean()), 3),
            "max_possible": TOPK,
            "frac_rows_using_6_distinct": round(float((distinct == TOPK).mean()), 4),
            "fetch_amplification_if_block_is_unit":
                round(float(distinct.mean()) * width / TOPK, 2),
        }
    out["contiguous_block_occupancy"] = occ
    return out


# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(prog="resident_sim.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", required=True, action="append")
    p.add_argument("--capacities", default="4500,5711,8000")
    p.add_argument("--modes", default="exact,verify-only,backfill")
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--t-layer-ms", type=float, default=2.0)
    p.add_argument("--t-io-ms", type=float, default=4.18)
    p.add_argument("--io-qd", type=int, default=8)
    p.add_argument("--nvme-gbps", type=float, default=4.5)
    p.add_argument("--backfill-per-token", type=float, default=19.2,
                   help="experts the drive lands per 80 ms step (4.5 GB/s / 18.8 MB)")
    p.add_argument("--queue-cap", type=int, default=2048)
    p.add_argument("--accepted-per-block", type=float, default=3.7)
    p.add_argument("--model-config", default=r"D:\models\DeepSeek-V4.1-Flash\config.json")
    p.add_argument("--out", default=None, help="directory for the JSON results")
    p.add_argument("--tag", default="resident_sim")
    args = p.parse_args()

    tr = load_trace(args.trace, args.max_tokens)
    caps = [int(c) for c in args.capacities.split(",") if c]
    modes = [m for m in args.modes.split(",") if m]

    results = []
    for cap in caps:
        for mode in modes:
            r = replay(tr, cap, mode, args)
            results.append(r)
            st = r.get("resident_only") or r.get("exact")
            print("[%-12s C=%5d] served %.4f  mass_lost %.5f  %.1f ms/tok  %.2f tok/s"
                  % (mode, cap, st["served_frac"], st["mean_mass_lost"],
                     r["ms_per_token"], r["tokens_per_s"]), flush=True)

    cfg = None
    try:
        with open(args.model_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except OSError:
        pass
    groups = group_analysis(tr, cfg)
    print("group-limited routing:", groups["group_limited_routing"])

    report = {"args": vars(args), "results": results, "groups": groups}
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, args.tag + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print("wrote", path)


if __name__ == "__main__":
    main()
