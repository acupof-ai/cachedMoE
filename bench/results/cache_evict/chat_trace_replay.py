# The ledger workload (8-turn chat route.bin): LRU vs E1's best vs Belady at 5,500 slots, cold start.
import sys, numpy as np
sys.path.insert(0, 'tools')
import cache_evict_study as c, hitrate_sim as h
step, pos, ids, hits = h.load_route(sys.argv[1])
n = ids.shape[0]
keys = (ids.astype(np.int32) + (np.arange(40, dtype=np.int32) * 384)[None, :, None]).reshape(n * 40, 6)
k16 = np.concatenate([keys, np.full((n * 40, 10), -1, np.int32)], 1); sc = np.ones((n * 40, 16), np.float32)
print(f"{n} steps ({n*40} rows); engine's own hit over the run {hits.sum()/(n*40*6):.4f}")
for pol, par in [('lru', {}), ('score-ext', dict(alpha=3200, decay=0.9, w_near=0.0, near_k=6)), ('belady', {})]:
    for cap in (5500,):
        r = c.replay(keys, k16, sc, cap, pol, par); print(f"{pol:10s} C={cap} hit {r['hit_rate']:.4f}  miss/token {r['miss_per_token']:.2f}")
