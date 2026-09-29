# Track E1 revisited: LRU extended by the key's demand touches within the last W tokens (STATUS §7 0ac).
# rank = last_use + alpha * (demand touches of this key within the last W tokens)
import sys, json, time, numpy as np
from collections import deque
from multiprocessing import Pool
sys.path.insert(0, 'tools')
import cache_evict_study as c

class WinExt(c.MinRank):
    def __init__(self, cap, alpha=1000.0, W=300, **kw):
        super().__init__(cap); self.alpha, self.W = alpha, W
        self.hist = {}; self.clock = 0; self.tok = 0
    def _demand(self, k, w):
        self.clock += 1; t = self.clock // 240          # ~240 demand touches a token
        h = self.hist.get(k)
        if h is None: h = self.hist[k] = deque()
        h.append(t)
        while h and h[0] < t - self.W: h.popleft()
        self._set(k, self.clock + self.alpha * len(h))
    def touch(self, k, w=1.0): self._demand(k, w)
    def admit(self, k, w=1.0):
        self._demand(k, w)
        if len(self.rank) > self.cap: self._evict()
    def _forget(self, k): self.hist.pop(k, None)
c.MAKE['win-ext'] = WinExt

Z = {}
def init(p):
    z = np.load(p); Z.update(k6=z['keys6'], k16=z['keys16'], sc=z['scores16'], pr=z['prompt'])
def job(a):
    pol, par, cap, warm = a
    r = c.replay(Z['k6'], Z['k16'], Z['sc'], cap, pol, par, warm_rows=warm)
    return pol, par, cap, warm, r['hit_rate'], r['tok_s']
if __name__ == '__main__':
    npz = sys.argv[1]; init(npz)
    # test split as E1 §12.5: prompts >= 28 are the test set, warmed by the rows before them
    pr = Z['pr']; test0 = int(np.nonzero(pr >= 28)[0][0]); warm0 = max(0, test0 - 2000 * 40)
    jobs = []
    for cap in (5100,):
        for pol, par in [('lru', {}), ('belady', {}), ('score-ext', dict(alpha=3200, decay=0.9, w_near=2.0, near_k=16))]:
            jobs.append((pol, par, cap, 0))
        for W in (30, 100, 300, 1000):
            for alpha in (500, 2000, 8000, 32000):
                jobs.append(('win-ext', dict(W=W, alpha=alpha), cap, 0))
    t0 = time.time()
    with Pool(24, initializer=init, initargs=(npz,)) as p:
        res = p.map(job, jobs)
    res.sort(key=lambda r: -r[4])
    for pol, par, cap, warm, hr, ts in res: print(f"{pol:10s} {json.dumps(par):48s} C={cap} hit {hr:.4f} tok/s {ts:.3f}")
    print(f"{time.time()-t0:.0f} s")
    json.dump([dict(policy=r[0], params=r[1], capacity=r[2], hit=r[4], tok_s=r[5]) for r in res], open(sys.argv[2], 'w'), indent=1)
