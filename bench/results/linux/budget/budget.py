"""Per-token time budget of a hitrate_bench run, with each part next to its floor.

    .venv/bin/python bench/results/linux/budget/budget.py RUN_DIR [RUN_DIR ...]
"""
import json, re, sys

UMA_GBS = 216.0          # measured LPDDR5X ceiling (kernel_p1.md §2.2)
DISK_GBS = 4.8           # internal SN740, deep-queue random 4 MiB (io_dst_bench)
EXPERT_MB = 18.808832    # one routed expert slot

for d in sys.argv[1:]:
    t = json.load(open(d + "/turns.json"))
    T = [x for x in t["turns"] if x.get("event") == "done"]
    n = sum(x["sampled_steps"] for x in T)
    avg = lambda k: sum(x["per_token_ms"].get(k, 0) * x["sampled_steps"] for x in T) / n
    steps = sum(x["decode_steps"] for x in T); dms = sum(x["decode_ms"] for x in T)
    hit = sum(x["decode_hit_rate"] * x["decode_steps"] for x in T) / steps
    wall = sum(x["total_ms"] for x in T) / 1000
    ttft = sum(x["ttft_ms"] for x in T) / 1000
    gen = sum(x["generated"] for x in T)
    mb = sum(x.get("decode_nvme_mb", 0) for x in T) / steps
    per_tok = dms / steps
    parts = {k: avg(k) for k in ("attn", "moe_gpu", "moe_host", "nvme_stall", "engram", "tail",
                                  "other", "record", "submit", "bind", "fence_wait", "sample_host")}
    misses = (1 - hit) * 240
    print(f"== {d}")
    print(f"decode {1000/per_tok:.3f} tok/s = {per_tok:.1f} ms/token | end-to-end {gen/wall:.3f} tok/s | "
          f"TTFT {ttft:.0f} s of {wall:.0f} s wall ({ttft/wall*100:.0f}%) | hit {hit:.4f}, {misses:.1f} misses/token, "
          f"{mb:.0f} MB read/token")
    for k, v in parts.items():
        print(f"   {k:12s} {v:7.2f} ms  ({v/per_tok*100:4.1f}%)")
    io_floor = misses * EXPERT_MB / 1000 / DISK_GBS * 1000
    print(f"   floors: NVMe for {misses:.1f} misses at {DISK_GBS} GB/s = {io_floor:.1f} ms "
          f"(measured stall {parts['nvme_stall']:.1f}); dense weights 8.52 GB at {UMA_GBS} GB/s = "
          f"{8522.85/UMA_GBS:.1f} ms (attn+tail+engram {parts['attn']+parts['tail']+parts['engram']:.1f})")
    st = t.get("status", {})
    gp = st.get("gate_probe", "") if isinstance(st, dict) else ""
    if gp: print("   " + gp.replace("\n", "\n   "))
    io = st.get("io", "") if isinstance(st, dict) else ""
    for line in io.splitlines():
        if line.strip().startswith(("P0:", "src[", "P0 ")): print("   io " + line.strip())
