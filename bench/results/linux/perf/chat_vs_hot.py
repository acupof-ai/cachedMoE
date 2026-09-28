"""Per-stage GPU busy of a chat trace's decode steps vs the hot step, and where the gaps are."""
import json, sys
from collections import defaultdict
sys.path.insert(0, "tools")
from trace_timeline import Trace

hot = json.load(open(sys.argv[2]))["hot"]["rows"]
hot_ms = {r["stage"]: r["ms"] for r in hot}
t = Trace(open(sys.argv[1], "rb").read())
steps = []
for tok in t.tokens():
    for p in t.passes(tok):
        recs = sorted([r for r in p if r.timed], key=lambda r: r.seq)
        if len(recs) < 700 or len(recs) > 900:      # decode steps only (prefill has other shapes)
            continue
        steps.append(recs)
n = len(steps)
busy = defaultdict(float); gap_before = defaultdict(float)
span = 0.0
for recs in steps:
    span += (recs[-1].end_ns - recs[0].begin_ns) / 1e6
    for prev, cur in zip(recs, recs[1:]):
        g = max(0, cur.begin_ns - prev.end_ns) / 1e6
        gap_before[t.name(cur)] += g
    for r in recs:
        busy[t.name(r)] += r.busy_ns / 1e6
tb = sum(busy.values()) / n; tg = sum(gap_before.values()) / n
print(f"{n} decode steps: GPU span {span/n:.2f} ms/token = busy {tb:.2f} + gaps {tg:.2f}")
print(f"{'stage':22s} {'chat':>7s} {'hot':>7s} {'delta':>7s} | {'gap in front':>12s}")
rows = sorted(set(busy) | set(gap_before), key=lambda s: -(busy[s] / n - hot_ms.get(s, 0) + gap_before[s] / n))
for s in rows[:16]:
    print(f"{s:22s} {busy[s]/n:7.2f} {hot_ms.get(s,0):7.2f} {busy[s]/n-hot_ms.get(s,0):7.2f} | {gap_before[s]/n:12.2f}")
