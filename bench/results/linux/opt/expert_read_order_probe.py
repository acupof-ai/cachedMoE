# One real expert read five ways (bench/probes/idle_gap_probe.py setup, 3 ms sleep between): manifest order, weights extent first, last chunk of each extent merged when <= 1.25 MiB, scales extent alone, weights extent alone.
import json, mmap, os, random, statistics, sys, time, threading
from pathlib import Path
sys.path.insert(0, 'bench/probes'); from idle_gap_probe import burst
model = Path.home() / 'models/DeepSeek-V4.1-Flash'
man = json.loads((model / 'deepmoe_manifest.json').read_text())
fds = [os.open(model / f['path'], os.O_RDONLY | os.O_DIRECT) for f in man['files']]
experts = [[(r['file'], r['aligned_off'], r['aligned_bytes']) for r in e]
           for l in man['experts'] if l['layer'] < 40 for e in l['experts']]
bufs = [mmap.mmap(-1, 2 << 20) for _ in range(8)]
def cut(ext, merge):
    out = []
    for off, n in ext:
        i = 0
        while i < n:
            c = n - i if merge and n - i < (1 << 20) * 5 // 4 else min(1 << 20, n - i)
            out.append((off + i, c)); i += c
    return out
def rd(fd, chunks):
    todo = list(chunks); lock = threading.Lock()
    def w(buf):
        while True:
            with lock:
                if not todo: return
                off, n = todo.pop(0)
            os.preadv(fd, [memoryview(buf)[:n]], off)
    t0 = time.perf_counter(); ts = [threading.Thread(target=w, args=(b,)) for b in bufs]
    [t.start() for t in ts]; [t.join() for t in ts]; return (time.perf_counter() - t0) * 1e3
V = {'manifest': lambda s, w: cut([s, w], False), 'weights_first': lambda s, w: cut([w, s], False),
     'merged_tails': lambda s, w: cut([s, w], True), 'scales_only': lambda s, w: cut([s], False),
     'weights_only': lambda s, w: cut([w], False)}
res = {k: [] for k in V}; rng = random.Random(2)
for _ in range(60):
    for k, f in V.items():
        e = rng.choice(experts); fd = fds[e[0][0]]
        time.sleep(0.003)
        res[k].append(rd(fd, f((e[0][1], e[0][2]), (e[1][1], e[1][2]))))
for k, v in res.items(): print(f'{k:14} median {statistics.median(v):.2f} ms  mean {statistics.mean(v):.2f}')
