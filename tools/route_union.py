# u(M): the expert union of M CONSECUTIVE positions, per layer, on real routings.
#
# docs/p4_dspark_runtime.md 6.5's byte account rests on this and had one measured
# point (26 at M=6, docs/p3_dspark.md 12.6). traces/mixed has the top-6 of every
# (prompt, position, layer), which is exactly what a verify batch of M
# consecutive positions would route to.
#
# Read-only over traces/mixed. No pandas (not installed); pyarrow only.
import sys, glob, os
import pyarrow.parquet as pq
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Asus\code\deepmoe\traces\mixed"
Ms = [2, 3, 4, 5, 6]
files = sorted(glob.glob(os.path.join(root, "route_layer*.parquet")))
if not files:
    sys.exit("no route_layer*.parquet under " + root)

# accumulate per M: sum of union sizes and count
tot = {m: [0, 0] for m in Ms}
per_layer = {}
for f in files:
    layer = int(os.path.basename(f)[11:13])
    t = pq.read_table(f, columns=["prompt_id", "pos", "top6_ids"])
    pid = t.column("prompt_id").to_numpy()
    pos = t.column("pos").to_numpy()
    ids = np.asarray(t.column("top6_ids").combine_chunks().flatten().to_numpy()).reshape(-1, 6)
    # order by (prompt, pos) so consecutive rows are consecutive positions
    order = np.lexsort((pos, pid))
    pid, pos, ids = pid[order], pos[order], ids[order]
    lay = {}
    for m in Ms:
        s = c = 0
        n = len(pos)
        # windows of m consecutive rows that are really consecutive positions of
        # one prompt
        ok = np.ones(n - m + 1, dtype=bool)
        for j in range(1, m):
            ok &= (pid[j:n - m + 1 + j] == pid[0:n - m + 1])
            ok &= (pos[j:n - m + 1 + j] == pos[0:n - m + 1] + j)
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            continue
        # sample at most 20k windows a layer a M; the estimate is already tight
        if idx.size > 20000:
            idx = idx[np.linspace(0, idx.size - 1, 20000).astype(np.int64)]
        for i in idx:
            s += len(np.unique(ids[i:i + m]))
            c += 1
        tot[m][0] += s
        tot[m][1] += c
        lay[m] = s / c
    per_layer[layer] = lay
    print(f"  layer {layer:2d}: " + "  ".join(f"u({m})={lay.get(m, 0):5.2f}" for m in Ms),
          flush=True)

print()
print("u(M) over all layers, real consecutive-position routings")
print(f"{'M':>2} {'u(M)':>7} {'u/6M':>7} {'linear 6+4(M-1)':>16}")
for m in Ms:
    s, c = tot[m]
    u = s / c
    print(f"{m:>2} {u:>7.2f} {u / (6 * m):>7.3f} {6 + 4 * (m - 1):>16}")
