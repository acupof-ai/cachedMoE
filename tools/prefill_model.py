"""The prefill counterpart of tools/gpu_model.py: what every prefill op SHOULD
cost from its FLOPs and bytes, next to what it DID cost.

    build/prefill_bench --section prefill --n 4133 --ids ids.txt --ops-json ops.jsonl
    python tools/prefill_model.py ops.jsonl [--n 4133] [--json out.json]

prefill_bench records each op's analytic work at dispatch, from the real
shapes (gpu/vulkan/prefill_kernels.h, PrefillTimes::Op): its FLOPs, and the
bytes it must move at least -- each input read once, each output written once,
nothing for an intermediate a fused kernel would keep on chip. The ceilings are
bench/probes/model_probe.cpp's (bench/results/linux/gpu_model/constants.json):

    floor = calls * t0 + max(flop / mma, bytes / dram)     a GPU op
    floor = max(bytes / disk, reads / iops)                an "io: " row

  mma    the fp16 cooperative-matrix rate with the tiles in registers
         (probe_mma.slang): every FLOP of the prefill is a dot product that can
         run on the matrix unit, so all are priced at it, the ones that run on
         the FMA units today included -- the floor is what a rewrite of the op
         could reach, not what its current kernel could
  dram   the best cold stream at any workgroup count
  t0     one dispatch alone (launch + drain)
  disk   one internal NVMe's continuous expert stream (--disk-gbs; the
         prefill's 4.66 GB/s, docs/STATUS.md §7)
  iops   the same drive's best 4 KiB random-read rate through the IoEngine
         (build/nvme_bench --chunk-kb 4 --pattern rand, --iops-csv): what the
         engram rows, one 4 KiB page per value or scale row, are bound by

"excess" = measured - floor: what the op itself wastes. An "io: " row's ms is
the wait the compute did NOT hide, so its excess can be negative (the read
overlapped); its floor is what the drive needs regardless.

What a fix WINS is a different number, because the prefill is two phases per
layer (gpu/vulkan/prefill_kernels.cpp run_layer / run_moe):

  serial  everything but the routed experts -- attention, the dense linears,
          the elementwise ops, the indexer, the engram, the shared expert, the
          host steps. The drive has nothing to read: a layer's experts are not
          known until its gate has run.
  moe     the routed experts, their reads double-buffered against their GPU
          work, so the phase is max(drive, GPU) plus the first batch's fill.

    prefill ~ serial + moe,   moe >= max(disk floor, routed GPU)

so an op of the serial phase wins its whole excess, the routed experts' GPU
work only what the phase spends above the drive's floor, and the one lever
the per-op table cannot show is the drive idling through the serial phase:
reads that ran during it would leave max(disk floor, serial + routed GPU).

Nesting: "gpu: attn ..." rows are GPU timestamps of the stages inside one
"attn (coop, submit+wait)" submit and are shown under it, not summed. The
block's own cost is the fused attention's (q, the index lists and the window
KV read once, o written once), so its excess is what fusing the six stages
could win; each stage's is what rewriting that stage alone could.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONSTANTS = ROOT / "bench" / "results" / "linux" / "gpu_model" / "constants.json"
IOPS_CSV = ROOT / "bench" / "results" / "linux" / "prefill_model" / "nvme_rand4k.csv"


def ceilings(path: Path, disk_gbs: float, iops_csv: Path) -> dict:
    c = json.loads(path.read_text())
    if "mma_f16_tflops" not in c:
        sys.exit(f"{path} has no mma_f16_tflops: rerun build/model_probe --json {path}")
    cold = [r for r in c["stream_cold"] if r["bytes"] >= 60e6]
    iops = max(float(r["iops"]) for r in csv.DictReader(iops_csv.open()) if r["chunk_kb"] == "4")
    return {"mma": c["mma_f16_tflops"] * 1e12, "dram": max(r["bytes"] / (r["us"] * 1e-6) for r in cold),
            "t0_s": c["dispatch_alone_us"] * 1e-6, "disk": disk_gbs * 1e9, "iops": iops,
            "fma": c["fma_tflops"] * 1e12}


def kind(op: str) -> str:
    if op.startswith("io: "):
        return "io"
    if op.startswith("host: "):
        return "host"
    if op.startswith("gpu: "):
        return "stage"
    return "gpu"


def model(op: dict, k: dict) -> dict:
    s, cls = op["ms"] / 1e3, kind(op["op"])
    t_mma, t_mem = op["flop"] / k["mma"], op["bytes"] / k["dram"]
    if cls == "io":
        t_bw, t_iops = op["bytes"] / k["disk"], op.get("reads", 0) / k["iops"]
        floor, bound = max(t_bw, t_iops), "disk" if t_bw >= t_iops else "iops"
    elif cls == "host" or (op["flop"] == 0 and op["bytes"] == 0):
        floor, bound = 0.0, "host" if cls == "host" else "?"
    else:
        # a stage inside one submit pays no launch of its own
        launch = 0.0 if cls == "stage" else op["calls"] * k["t0_s"]
        floor = launch + max(t_mma, t_mem)
        bound = "mma" if t_mma >= t_mem else "dram"
        if max(t_mma, t_mem) < launch:
            bound = "launch"
    return {**op, "cls": cls, "floor_ms": floor * 1e3, "excess_ms": (s - floor) * 1e3, "bound": bound,
            "tflops": op["flop"] / s / 1e12 if s > 0 else 0.0, "gbs": op["bytes"] / s / 1e9 if s > 0 else 0.0}


def group(op: str) -> str:
    if op.startswith("io: "):
        return op
    if op.startswith("host: "):
        return "host"
    if op.startswith("attn") or op.startswith("prefill_attn s0") or op.startswith("prefill_attn s1"):
        return "band attention"
    if op.startswith("prefill_attn s2"):
        return "indexer scores"
    if op.startswith("moe "):
        return op
    if op.startswith("gemm ") or op.startswith("coop "):
        return "dense linears"
    return "elementwise"


def report(run: dict, k: dict, top: int) -> dict:
    rows = [model(o, k) for o in run["ops"]]
    top_rows = [r for r in rows if r["cls"] != "stage"]
    stages = [r for r in rows if r["cls"] == "stage"]
    top_rows.sort(key=lambda r: -r["excess_ms"])
    gpu = [r for r in top_rows if r["cls"] == "gpu"]
    io = [r for r in top_rows if r["cls"] == "io"]
    total = run["total_ms"]
    covered = sum(r["ms"] for r in top_rows)
    gpu_floor = sum(r["floor_ms"] for r in gpu)
    disk_floor = sum(r["floor_ms"] for r in io)

    print(f"== N={run['n']} {run['mode']}: prefill {total / 1e3:.1f} s; ops cover {covered / 1e3:.1f} s "
          f"({len(top_rows)} rows, {run['dispatches']} dispatches in {run['submits']} submits)")
    print(f"   ceilings: matrix {k['mma'] / 1e12:.1f} TFLOP/s (fp32 FMA {k['fma'] / 1e12:.1f}), DRAM "
          f"{k['dram'] / 1e9:.0f} GB/s, launch {k['t0_s'] * 1e6:.1f} us, disk {k['disk'] / 1e9:.2f} GB/s "
          f"and {k['iops'] / 1e3:.0f}K 4 KiB reads/s")
    flop = sum(r["flop"] for r in gpu)
    byt = sum(r["bytes"] for r in gpu)
    print(f"   work: {flop / 1e12:.1f} TFLOP and {byt / 1e9:.0f} GB on the GPU, "
          f"{sum(r['bytes'] for r in io) / 1e9:.1f} GB from disk")
    print(f"   floor: GPU {gpu_floor / 1e3:.2f} s (matrix {flop / k['mma']:.2f} s, DRAM {byt / k['dram']:.2f} s, "
          f"summed per op), disk {disk_floor / 1e3:.2f} s -> overlapped, the prefill cannot be under "
          f"{max(gpu_floor, disk_floor) / 1e3:.2f} s; measured {total / 1e3:.1f} s = "
          f"{total / max(gpu_floor, disk_floor, 1e-9):.0f}x")
    hdr = (f"   {'op':34s} {'calls':>6s} {'ms':>9s} {'GFLOP':>9s} {'GB':>8s} {'TFLOP/s':>7s} {'GB/s':>6s} "
           f"{'floor ms':>9s} {'x':>6s} {'excess ms':>9s} bound")
    print(hdr)

    def line(r: dict, indent: str = "") -> None:
        x = r["ms"] / r["floor_ms"] if r["floor_ms"] > 0 else float("inf")
        print(f"   {indent + r['op']:34.34s} {r['calls']:6d} {r['ms']:9.1f} {r['flop'] / 1e9:9.1f} "
              f"{r['bytes'] / 1e9:8.2f} {r['tflops']:7.2f} {r['gbs']:6.1f} {r['floor_ms']:9.1f} "
              f"{x:6.1f} {r['excess_ms']:9.1f} {r['bound']}")

    for r in top_rows[:top]:
        line(r)
        if r["op"].startswith("attn (coop"):
            for st in sorted(stages, key=lambda s: -s["excess_ms"]):
                line(st, "  ")
    rest = top_rows[top:]
    if rest:
        print(f"   ({len(rest)} more rows: {sum(r['ms'] for r in rest):.0f} ms, "
              f"{sum(r['excess_ms'] for r in rest):.0f} ms excess)")
    by = defaultdict(lambda: [0.0, 0.0, 0.0])
    for r in top_rows:
        g = by[group(r["op"])]
        g[0] += r["ms"]; g[1] += r["floor_ms"]; g[2] += r["excess_ms"]
    print("   by group (ms / floor / excess): " + "; ".join(
        f"{name} {v[0]:.0f} / {v[1]:.0f} / {v[2]:.0f}" for name, v in sorted(by.items(), key=lambda kv: -kv[1][2])))

    # the critical path: the serial phase, then the routed experts' phase
    moe_rows = [r for r in top_rows if r["op"] in ("io: experts", "moe routed (gpu)")]
    moe_ms = sum(r["ms"] for r in moe_rows)
    need = sum(r["floor_ms"] for r in moe_rows if r["cls"] == "io")
    moe_gpu = sum(r["ms"] for r in moe_rows if r["cls"] == "gpu")
    serial = total - moe_ms
    wins = {name: v[2] for name, v in by.items() if name not in ("io: experts", "moe routed (gpu)")}
    wins["moe routed (gpu)"] = max(0.0, min(by["moe routed (gpu)"][2], moe_ms - need))
    wins["expert reads during the serial phase"] = max(0.0, total - max(need, serial + moe_gpu))
    print(f"   critical path: serial {serial / 1e3:.1f} s (the drive idle) + routed experts {moe_ms / 1e3:.1f} s "
          f"(the drive needs {need / 1e3:.1f}, their GPU work {moe_gpu / 1e3:.1f})")
    print("   what each fix wins on that path (s): " + ", ".join(
        f"{name} {v / 1e3:.1f}" for name, v in sorted(wins.items(), key=lambda kv: -kv[1]) if v >= 50))
    return {"n": run["n"], "mode": run["mode"], "total_ms": total, "covered_ms": covered,
            "gpu_floor_ms": gpu_floor, "disk_floor_ms": disk_floor, "rows": top_rows, "stages": stages,
            "groups": {name: {"ms": v[0], "floor_ms": v[1], "excess_ms": v[2]} for name, v in by.items()},
            "serial_ms": serial, "moe_ms": moe_ms, "moe_disk_ms": need, "moe_gpu_ms": moe_gpu, "wins_ms": wins}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ops", type=Path, help="prefill_bench --ops-json output (one JSON line a run)")
    ap.add_argument("--n", type=int, help="only the runs of this prompt length")
    ap.add_argument("--constants", type=Path, default=CONSTANTS)
    ap.add_argument("--disk-gbs", type=float, default=4.66)
    ap.add_argument("--iops-csv", type=Path, default=IOPS_CSV)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    k = ceilings(a.constants, a.disk_gbs, a.iops_csv)
    runs = [json.loads(line) for line in a.ops.read_text().splitlines() if line.strip()]
    out = [report(r, k, a.top) for r in runs if a.n is None or r["n"] == a.n]
    if a.json:
        a.json.write_text(json.dumps({"ceilings": k, "runs": out}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
