"""What the cache knows at the moment it evicts (docs/STATUS.md §7 0ac).

Replays LRU over the routing trace and, at a sample of its evictions, looks at
the oldest candidates: how many Belady would have kept (needed again within
100 tokens), and how well each history feature -- age, touches in the last 30
or 300 tokens, lifetime touches -- separates those from the rest (AUC). This is
the information ceiling of any online eviction policy on this trace; E1's
+2% is what that ceiling is worth.

    .venv/bin/python tools/evict_tail_info.py bench/results/cache_evict/trace_cols.npz

(the npz is cache_evict_study.py's column cache of traces/mixed).
"""
import sys, numpy as np
from collections import OrderedDict, deque
z = np.load(sys.argv[1]); k6 = z['keys6']; rows = k6.shape[0]
key = k6.ravel().astype(np.int64); n = len(key); tok = np.arange(n) // 240
order = np.argsort(key, kind='stable'); s = key[order]
nxt = np.full(n, -1); same = s[1:] == s[:-1]
no = np.where(np.r_[same, False], np.r_[order[1:], -1], -1); nxt[order] = no
INF = 10**9
CAP = 5100; CAND = 400; EVERY = 40
cache = OrderedDict()       # key -> last access index
hist = {}                   # key -> deque of touch tokens (last 300)
tot = {}                    # key -> lifetime touches
def auc(score, y):
    r = np.argsort(np.argsort(score, kind='stable')) + 1.0
    p = y.sum(); q = len(y) - p
    return (r[y].sum() - p * (p + 1) / 2) / (p * q) if p and q else np.nan
S_age, S_c300, S_c30, S_tot, Y, regret, ev = [], [], [], [], [], [], 0
for g in range(n):
    k = int(key[g]); t = g // 240
    h = hist.get(k)
    if h is None: h = hist[k] = deque()
    h.append(t)
    while h[0] < t - 300: h.popleft()
    tot[k] = tot.get(k, 0) + 1
    if k in cache:
        cache.move_to_end(k); cache[k] = g; continue
    cache[k] = g
    if len(cache) > CAP:
        ev += 1
        if ev % EVERY == 0:
            cand = list(cache.items())[:CAND]
            ks = np.array([c[0] for c in cand]); last = np.array([c[1] for c in cand])
            dn = np.array([(tok[nxt[l]] - t) if nxt[l] >= 0 else INF for l in last], float)
            # Belady keeps the soon ones; label = "needed within the median horizon of the candidates"
            H = 100
            y = dn <= H
            c300 = np.array([sum(1 for x in hist[int(kk)] if x >= t - 300) for kk in ks], float)
            c30 = np.array([sum(1 for x in hist[int(kk)] if x >= t - 30) for kk in ks], float)
            S_age.append(auc(last.astype(float), y)); S_c300.append(auc(c300, y)); S_c30.append(auc(c30, y))
            S_tot.append(auc(np.array([tot[int(kk)] for kk in ks], float), y)); Y.append(y.mean())
            regret.append((dn[0], dn.max()))
        cache.popitem(last=False)
Y = np.array(Y); R = np.array(regret)
print(f"evictions {ev:,}, sampled {len(Y)}; among the {CAND} oldest cached keys, {100*Y.mean():.1f}% are needed within 100 tokens")
print(f"LRU victim needed again after median {np.median(R[:,0]):.0f} tokens; the key Belady would drop: {np.median(np.minimum(R[:,1], 1e6)):.0f} (INF = never) ; victim never reused {100*(R[:,0]>=INF).mean():.0f}%")
for nm, v in (('age (LRU)', S_age), ('count last 30', S_c30), ('count last 300', S_c300), ('lifetime count', S_tot)):
    v = np.array(v); print(f"  AUC at eviction, {nm:15s}: {np.nanmean(v):.3f}")
