"""moe_down / late-gateup busy time binned by how long the GPU sat idle before the layer's MoE."""
import sys
from collections import defaultdict
sys.path.insert(0, "tools")
from trace_timeline import Trace
t = Trace(open(sys.argv[1], "rb").read())
bins = [0, 0.05, 0.5, 1, 2, 4, 8, 1e9]
b = defaultdict(list)
for tok in t.tokens():
    for p in t.passes(tok):
        recs = sorted([r for r in p if r.timed], key=lambda r: r.seq)
        if not (700 <= len(recs) <= 900):
            continue
        for i in range(2, len(recs)):
            if t.name(recs[i]) != "moe_down":
                continue
            a = recs[i - 1]
            gap = max(0, a.begin_ns - recs[i - 2].end_ns) / 1e6
            k = next(j for j in range(len(bins) - 1) if bins[j] <= gap < bins[j + 1])
            b[k].append((recs[i].busy_ns / 1e3, a.busy_ns / 1e3))
print("idle before the MoE (ms)      n   gateup p50   down p50  down mean")
for k in sorted(b):
    v = b[k]; d = sorted(x[0] for x in v); g = sorted(x[1] for x in v)
    print(f"  [{bins[k]:5.2f},{bins[k+1]:>7g})  {len(v):6d}  {g[len(g)//2]:10.1f} {d[len(d)//2]:10.1f} {sum(d)/len(d):10.1f}")
