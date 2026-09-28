"""One report of where a decode token's time goes, each part next to its floor.

    # hot step (every expert resident, no NVMe): capture a trace and report
    python tools/perf_report.py --capture

    # an existing trace, plus a chat run's end-to-end budget
    python tools/perf_report.py --trace t.bin --chat bench/results/linux/stripe/2_stripe1

The hot step says what the GPU spends per token when nothing is read from disk:
every dispatch stage's busy time against the bytes of weights it has to stream,
at the measured memory ceiling (`--bw`, GB/s). A stage's floor is bytes / bw;
its EXCESS is busy - floor, and the stages are ranked by excess because that is
the time an optimisation of that stage could at most win. Stages that read no
weights (softmax, top-k, norms) have a floor of ~0 and show up by their busy
time. Barriers between dispatches and host time not covered by the GPU are
reported on their own lines.

The chat part (`--chat RUN_DIR`, a tools/hitrate_bench.py output) puts the hot
step next to a real conversation: compute per token, the NVMe stall against its
own floor (misses x expert bytes / disk GB/s), and the host-side rest.

Reading it: one measurement is enough to see a change in a single stage -- a
stage's busy time over 40 layers moves far less than end-to-end tok/s does. Use
the same --warm and the same state on both sides of a change.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from trace_timeline import Trace  # noqa: E402
import provenance  # noqa: E402

EXPERT_MAT = 5_898_240 + 368_640          # model/layout.h: one fp4 matrix + its scales
EXPERT_SLOT = 18_808_832                  # one routed expert as read from disk
TOPK = 6
LEDGER = ROOT / "bench" / "results" / "perf_ledger.jsonl"
DEFAULT_MODEL = (os.environ.get("DEEPMOE_MODEL_DIR")
                 or os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))

# stage name -> the manifest tensor prefixes (under layers.L.) it streams.
STAGE_TENSORS = {
    "wq_a": ["attn.wq_a."],
    "wq_b": ["attn.wq_b."],
    "wkv.gemv": ["attn.wkv."],
    "wo_a": ["attn.wo_a."],
    "wo_b": ["attn.wo_b."],
    "mega_mhc.mix": ["hc_attn_fn"],
    "mega_mhc.mix.ffn": ["hc_ffn_fn"],
    "gate.score": ["ffn.gate."],
    "compressor.wkv": ["attn.compressor.wkv."],
    "compressor.wgate": ["attn.compressor.wgate."],
    "indexer.wq_b": ["attn.indexer.wq_b."],
    "indexer.weights": ["attn.indexer.weights_proj."],
    "indexer.key": ["attn.indexer.wk."],
    "moe_shared_early": ["ffn.shared_experts.w1.", "ffn.shared_experts.w3."],
    "engram_gemv+gate": ["engram.wkv.", "engram.wgate.", "engram.gate."],
}


def load_manifest(model_dir: str) -> dict[str, int]:
    m = json.load(open(os.path.join(model_dir, "deepmoe_manifest.json")))
    ts = m.get("tensors", m)
    items = ts.items() if isinstance(ts, dict) else ((t["name"], t) for t in ts)
    return {name: int(t.get("bytes", 0)) for name, t in items}


def layer_bytes(man: dict[str, int], layer: int, prefixes: list[str]) -> int:
    base = f"layers.{layer}."
    return sum(b for n, b in man.items()
               if n.startswith(base) and any(n[len(base):].startswith(p) for p in prefixes))


def stage_bytes(man: dict[str, int], name: str, layer: int, has_se: bool) -> int:
    """Weight bytes stage `name` streams at `layer` for one decode token."""
    if name in STAGE_TENSORS:
        return layer_bytes(man, layer, STAGE_TENSORS[name])
    if name == "moe_gateup":
        b = TOPK * 2 * EXPERT_MAT
        if not has_se:
            b += layer_bytes(man, layer, ["ffn.shared_experts.w1.", "ffn.shared_experts.w3."])
        return b
    if name == "moe_down":
        return TOPK * EXPERT_MAT + layer_bytes(man, layer, ["ffn.shared_experts.w2."])
    if name == "head":
        return man.get("head.weight", 0)
    return 0


def capture(model_dir: str, warm: int, out: Path, extra_env: dict[str, str]) -> None:
    exe = ROOT / "build" / ("deepmoe.exe" if os.name == "nt" else "deepmoe")
    cmd = [str(exe), "run", "--model", model_dir, "--steps", "1", "--warm", str(warm),
           "--trace", str(out)]
    env = dict(os.environ, **extra_env)
    print("capturing:", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
        raise SystemExit(f"deepmoe run failed ({r.returncode})")


def hot_step(trace_path: Path, man: dict[str, int], bw: float, last: int = 10) -> dict:
    """Median over the trace's last `last` passes (the first ones warm the cache).

    One pass moves by a few % with the chip's temperature and clock; each
    stage's median over ten passes of the same token does not, and the spread
    of the span over those passes is reported so a reader can see it.
    """
    t = Trace(trace_path.read_bytes())
    tok = t.tokens()[0]
    passes = t.passes(tok)
    use = passes[-min(last, max(1, len(passes) - 1)):]
    per_stage = defaultdict(list)
    meta = {}
    spans, busys, gapss = [], [], []
    has_se = False
    for p in use:
        timed = [r for r in sorted(p, key=lambda r: r.seq) if r.timed]
        has_se = has_se or any(t.name(r) == "moe_shared_early" for r in timed)
        acc = defaultdict(int)
        for r in timed:
            name = t.name(r)
            acc[name] += r.busy_ns
            if name not in meta:
                meta[name] = {"n": 0, "bytes": 0, "cls": {0: "attention", 1: "ced", 2: "moe", 3: "engram",
                                                          4: "tail"}.get(r.cls, "other"), "seen": id(p)}
            if meta[name]["seen"] == id(p):
                meta[name]["n"] += 1
                layer = r.layer if r.layer != 0xFFFF else -1
                meta[name]["bytes"] += stage_bytes(man, name, layer, has_se)
        for name, v in acc.items():
            per_stage[name].append(v)
        spans.append(timed[-1].end_ns - timed[0].begin_ns if timed else 0)
        busys.append(sum(r.busy_ns for r in timed))
        gapss.append(sum(max(0, c.begin_ns - q.end_ns) for q, c in zip(timed, timed[1:])))
    med = lambda xs: sorted(xs)[len(xs) // 2]
    rows = []
    for name, vs in per_stage.items():
        m = meta[name]
        ms = med(vs) / 1e6
        floor = m["bytes"] / (bw * 1e9) * 1e3
        rows.append({"stage": name, "cls": m["cls"], "n": m["n"], "ms": ms,
                     "mb": m["bytes"] / 1e6, "floor": floor, "excess": ms - floor,
                     "gbs": (m["bytes"] / 1e9) / (ms / 1e3) if ms > 0 and m["bytes"] else 0.0})
    rows.sort(key=lambda x: -x["excess"])
    span, busy, gaps = med(spans), med(busys), med(gapss)
    return {"rows": rows, "busy": busy / 1e6, "span": span / 1e6, "gaps": gaps / 1e6,
            "host": max(0.0, (span - busy - gaps) / 1e6), "passes": len(passes), "used": len(use),
            "span_min": min(spans) / 1e6, "span_max": max(spans) / 1e6,
            "dispatches": sum(m["n"] for m in meta.values()), "has_se": has_se}


def chat_budget(run_dir: Path, disk_gbs: float) -> dict:
    t = json.load(open(run_dir / "turns.json"))
    T = [x for x in t["turns"] if x.get("event") == "done"]
    n = sum(x["sampled_steps"] for x in T)
    avg = lambda k: sum(x["per_token_ms"].get(k, 0) * x["sampled_steps"] for x in T) / n
    steps = sum(x["decode_steps"] for x in T)
    dms = sum(x["decode_ms"] for x in T)
    hit = sum(x["decode_hit_rate"] * x["decode_steps"] for x in T) / steps
    wall = sum(x["total_ms"] for x in T) / 1000
    gen = sum(x["generated"] for x in T)
    misses = (1 - hit) * TOPK * 40
    # NVMe stall against the step's own miss count, over the decode steps
    # (hit > 0.8; the cold first steps of a session are prefill-like). The slope
    # is ms per missed expert: across same-config runs it moves by ~1% where
    # tok/s moves by ~3%, so it is the number to compare for any I/O change.
    slope = icpt = None
    pf = run_dir / "profile.jsonl"
    if pf.exists():
        pts = [(r["expert_misses"], r["nvme_stall_ms"]) for r in map(json.loads, open(pf))
               if r.get("hit_rate", 0) > 0.8]
        if len(pts) > 10:
            mx = sum(x for x, _ in pts) / len(pts)
            my = sum(y for _, y in pts) / len(pts)
            sxx = sum((x - mx) ** 2 for x, _ in pts)
            if sxx:
                slope = sum((x - mx) * (y - my) for x, y in pts) / sxx
                icpt = my - slope * mx
    prefill_s = sum(x.get("prefill_ms", 0) for x in T) / 1000
    return {"per_tok": dms / steps, "tok_s": 1000 * steps / dms, "e2e": gen / wall, "hit": hit,
            "misses": misses, "stall": avg("nvme_stall"), "attn": avg("attn"),
            "moe_gpu": avg("moe_gpu"), "tail": avg("tail"), "engram": avg("engram"),
            "other": avg("other"), "moe_host": avg("moe_host"),
            "io_floor": misses * EXPERT_SLOT / (disk_gbs * 1e9) * 1e3,
            "stall_per_miss": slope, "stall_icpt": icpt, "prefill_s": prefill_s}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path, help="an existing `deepmoe run --trace` file")
    ap.add_argument("--capture", action="store_true", help="run `deepmoe run --warm` and trace it")
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for --capture")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--bw", type=float, default=230.0,
                    help="memory ceiling for the floors, GB/s (x: wq_b streams at ~229)")
    ap.add_argument("--chat", type=Path, help="a tools/hitrate_bench.py run directory")
    ap.add_argument("--disk-gbs", type=float, default=4.8,
                    help="aggregate disk rate for the stall floor (one SN740: 4.8; +mirror: ~8.5)")
    ap.add_argument("--top", type=int, default=14)
    ap.add_argument("--json", type=Path, help="also write the numbers here")
    ap.add_argument("--record", metavar="LABEL",
                    help="append this report, with the run's provenance, to --ledger "
                         "(docs/STATUS.md's measured table is rendered from it: tools/perf_ledger.py)")
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    a = ap.parse_args()

    man = load_manifest(a.model)
    out = {}
    trace = a.trace
    hot_prov = None
    if a.capture:
        trace = Path(tempfile.mkdtemp()) / "hot.bin"
        env = dict(kv.split("=", 1) for kv in a.env)
        hot_prov = provenance.capture(env=dict(os.environ, **env))
        capture(a.model, a.warm, trace, env)
    if trace:
        h = hot_step(trace, man, a.bw)
        out["hot"] = h
        floor = sum(r["floor"] for r in h["rows"])
        print(f"== hot step ({trace.name}, median of the last {h['used']} of {h['passes']} passes, "
              f"{h['dispatches']} dispatches, shared-early {'on' if h['has_se'] else 'off'})")
        print(f"   GPU span {h['span']:.2f} ms (passes {h['span_min']:.2f}..{h['span_max']:.2f}) = busy {h['busy']:.2f} + barriers {h['gaps']:.2f}"
              f" + host-late {h['host']:.2f};  weight floor at {a.bw:.0f} GB/s: {floor:.2f} ms"
              f"  -> {h['span'] - floor:.2f} ms above the floor")
        print(f"   {'stage':22s} {'cls':9s} {'ms/tok':>7s} {'MB':>8s} {'GB/s':>6s} "
              f"{'floor':>6s} {'excess':>7s} {'%ceil':>6s}")
        for r in h["rows"][:a.top]:
            pct = 100 * r["floor"] / r["ms"] if r["ms"] and r["floor"] else 0
            print(f"   {r['stage']:22s} {r['cls']:9s} {r['ms']:7.2f} {r['mb']:8.1f} "
                  f"{r['gbs']:6.0f} {r['floor']:6.2f} {r['excess']:7.2f} {pct:5.0f}%")
        rest = h["rows"][a.top:]
        if rest:
            print(f"   ({len(rest)} more stages: {sum(r['ms'] for r in rest):.2f} ms busy, "
                  f"{sum(r['excess'] for r in rest):.2f} excess)")
        no_w = sum(r["ms"] for r in h["rows"] if r["mb"] == 0)
        print(f"   stages that read no weights: {no_w:.2f} ms;  barriers: {h['gaps']:.2f} ms")
    if a.chat:
        c = chat_budget(a.chat, a.disk_gbs)
        out["chat"] = c
        compute = c["attn"] + c["moe_gpu"] + c["tail"] + c["engram"]
        print(f"\n== chat ({a.chat})")
        print(f"   decode {c['tok_s']:.3f} tok/s = {c['per_tok']:.1f} ms/token, end-to-end "
              f"{c['e2e']:.3f} tok/s, hit {c['hit']:.4f} ({c['misses']:.1f} misses/token)")
        print(f"   compute {compute:.1f} ms (attn {c['attn']:.1f} moe {c['moe_gpu']:.1f} "
              f"tail {c['tail']:.1f} engram {c['engram']:.1f})  NVMe stall {c['stall']:.1f} ms "
              f"(floor {c['io_floor']:.1f} at {a.disk_gbs} GB/s)  other {c['other']:.1f}")
        if c["stall_per_miss"] is not None:
            print(f"   stall per missed expert {c['stall_per_miss']:.2f} ms (+{c['stall_icpt']:.2f} ms/step);"
                  f"  one expert at {a.disk_gbs} GB/s: {EXPERT_SLOT / a.disk_gbs / 1e6:.2f} ms;"
                  f"  prefill total {c['prefill_s']:.1f} s")
        if "hot" in out:
            h = out["hot"]
            print(f"   vs hot step: compute in chat {compute:.1f} vs GPU span {h['span']:.1f} ms "
                  f"-> {compute - h['span']:.1f} ms of it is the chat's own (gate waits, cold L2)")
    if "hot" in out or "chat" in out:
        print("\n== where the time above the floors is (largest first)")
        items = []
        if "hot" in out:
            h = out["hot"]
            for r in h["rows"]:
                items.append((r["excess"], f"kernel {r['stage']} ({r['cls']})"))
            items.append((h["gaps"] + h["host"], "barriers + host-late on the hot step"))
        if "chat" in out:
            c = out["chat"]
            items.append((c["stall"] - c["io_floor"], "NVMe stall above its bandwidth floor"))
            items.append((c["io_floor"], "NVMe floor itself (fewer misses: cache size / hit)"))
        for v, what in sorted(items, key=lambda x: -x[0])[:10]:
            print(f"   {v:7.2f} ms  {what}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, default=str))
    if a.record:
        rec = {"label": a.record, "recorded": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "bw": a.bw, "disk_gbs": a.disk_gbs}
        if "chat" in out:
            rec["chat_dir"] = str(a.chat)
            rec["chat"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["chat"].items()}
            t = json.loads((a.chat / "turns.json").read_text())
            rec["run_cmd"], rec["run_env"] = t.get("cmd"), t.get("env")
            pf = a.chat / "provenance.json"
            if pf.exists():
                rec["provenance"] = json.loads(pf.read_text())
            else:
                # A run from before provenance.json existed: what the tree and the
                # binaries are NOW, flagged, so nobody mistakes it for a record.
                rec["provenance"] = dict(provenance.capture(), after_the_fact=True)
        if "hot" in out:
            h = out["hot"]
            rec["trace"] = str(trace)
            rec["hot"] = {"span": round(h["span"], 3), "busy": round(h["busy"], 3),
                          "span_range": [round(h["span_min"], 3), round(h["span_max"], 3)],
                          "passes_used": h["used"],
                          "barriers": round(h["gaps"], 3), "host_late": round(h["host"], 3),
                          "floor": round(sum(r["floor"] for r in h["rows"]), 3),
                          "stages": [{k: (round(r[k], 3) if isinstance(r[k], float) else r[k])
                                      for k in ("stage", "cls", "ms", "floor", "excess")}
                                     for r in h["rows"][:a.top]]}
            if hot_prov:
                rec.setdefault("provenance", hot_prov)
            rec.setdefault("provenance", dict(provenance.capture(), after_the_fact=True))
        a.ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(a.ledger, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        print(f"\nrecorded as '{a.record}' in {a.ledger}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
