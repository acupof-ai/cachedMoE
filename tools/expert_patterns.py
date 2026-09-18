import json, os, sys, math
import numpy as np
REPO = r"C:\Users\Asus\code\deepmoe\build\p4-integ"
sys.path.insert(0, os.path.join(REPO, "tools"))
import hitrate_sim  # noqa: E402

L, K, E = 40, 6, 384
GROUPS = {
    "window L0-1": list(range(0, 2)),
    "ratio2 L2-19": list(range(2, 20)),
    "ratio1 L20-39": list(range(20, 40)),
}
SOURCES = [2, 8, 14, 20]
INDEX_OWN = [2, 8, 14, 20, 24, 28, 32, 36]
TARGET = [37, 38, 39]

def load_run(d):
    step, pos, ids, hits = hitrate_sim.load_route(os.path.join(d, "route.bin"))
    turns = json.load(open(os.path.join(d, "turns.json"), encoding="utf-8"))
    kinds = hitrate_sim.step_kinds(turns["turns"])
    n = min(len(ids), len(kinds))
    ids, hits, step, pos, kinds = ids[:n], hits[:n], step[:n], pos[:n], kinds[:n]
    dec = np.array([k[0] == "decode" for k in kinds])
    turn = np.array([k[1] for k in kinds])
    return dict(step=step, pos=pos, ids=ids, hits=hits, dec=dec, turn=turn, server=turns["server"])

def per_layer(run):
    d = run["dec"]
    rows = []
    for l in range(L):
        hit = run["hits"][d, l].sum() / (K * max(1, d.sum()))
        ids = run["ids"][d, l].ravel()
        cnt = np.bincount(ids, minlength=E).astype(float)
        order = np.sort(cnt)[::-1]
        tot = order.sum()
        top1 = order[:1].sum() / tot if tot else 0
        top4 = order[:4].sum() / tot if tot else 0
        top16 = order[:16].sum() / tot if tot else 0
        uniq = int((cnt > 0).sum())
        rows.append(dict(layer=l, hit=float(hit), top1=float(top1), top4=float(top4),
                         top16=float(top16), unique=uniq, total=int(tot)))
    return rows

def group_stat(rows, layers):
    h = np.mean([rows[l]["hit"] for l in layers])
    t16 = np.mean([rows[l]["top16"] for l in layers])
    u = np.mean([rows[l]["unique"] for l in layers])
    return dict(hit=float(h), top16=float(t16), unique=float(u))

def locality(run, ks=(1, 4, 16, 128)):
    out = {}
    ids, dec = run["ids"], run["dec"]
    for k in ks:
        vals = []
        for l in range(L):
            active = np.zeros(E, dtype=bool)
            hist = []
            hit = tot = 0
            for t in range(len(ids)):
                if not dec[t]:
                    continue
                row = ids[t, l]
                hit += int(active[row].sum()); tot += K
                hist.append(row.copy())
                for x in row: active[x] = True
                if len(hist) > k:
                    old = hist.pop(0)
                    for x in old: active[x] = False
            vals.append(hit / max(1, tot))
        out[k] = float(np.mean(vals))
    return out

def turn_stability(run):
    out = {}
    turns = sorted(set(int(t) for t in run["turn"]))
    freqs = {}
    for t in turns:
        m = run["turn"] == t
        if m.sum() == 0: continue
        f = np.zeros((L, E))
        for l in range(L):
            f[l] = np.bincount(run["ids"][m, l].ravel(), minlength=E)
        freqs[t] = f
    cos_adj = []
    for a, b in zip(turns, turns[1:]):
        if a in freqs and b in freqs:
            x, y = freqs[a][freqs[a] > 0], freqs[b][freqs[a] > 0]
            if x.sum() and y.sum():
                cos_adj.append(float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))))
    out["adjacent_cos"] = float(np.mean(cos_adj)) if cos_adj else 0.0
    allf = sum(freqs.values())
    out["cos_first_last"] = 0.0
    if len(turns) >= 2 and turns[0] in freqs and turns[-1] in freqs:
        a, b = freqs[turns[0]].ravel(), freqs[turns[-1]].ravel()
        out["cos_first_last"] = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return out

def position_bins(run):
    pos = run["pos"]; dec = run["dec"]
    edges = [0, 128, 512, 2048, 8192, 1 << 30]
    out = []
    for lo, hi in zip(edges, edges[1:]):
        m = dec & (pos >= lo) & (pos < hi)
        if m.sum():
            out.append((f"{lo}-{hi-1}", int(m.sum()), float(run["hits"][m].sum() / (K * L * m.sum()))))
    return out

def heat_stats(path):
    counts = np.zeros((L, E))
    rows = 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("//"): continue
        if not line.startswith("{"): continue
        parts = line.strip("{},").split(",")
        if len(parts) < 3: continue
        l, e, c = int(parts[0]), int(parts[1]), int(parts[2])
        if 0 <= l < L and 0 <= e < E:
            counts[l, e] = c; rows += 1
    return counts, rows

def summarize(name, d):
    run = load_run(d)
    rows = per_layer(run)
    s = dict(name=name, server=run["server"].get("cache_slots"),
             steps=int(len(run["ids"])), decode_steps=int(run["dec"].sum()),
             overall_hit=float(run["hits"][run["dec"]].sum() / (K * L * run["dec"].sum())),
             groups={g: group_stat(rows, ls) for g, ls in GROUPS.items()},
             locality=locality(run), turn_stability=turn_stability(run),
             position_bins=position_bins(run), per_layer=rows,
             prefill_reuse=prefill_decode_reuse(run), persistent=persistent_experts(run))
    return s


def prefill_decode_reuse(run):
    """Share of decode selections whose expert already appeared in the prompt prefill."""
    pre = ~run["dec"]
    if pre.sum() == 0:
        return {"prefill_hit": 0.0, "decode_seen_in_prefill": 0.0, "prefill_steps": 0}
    ph = float(run["hits"][pre].sum() / (K * L * pre.sum()))
    seen = [set(run["ids"][pre, l].ravel().tolist()) for l in range(L)]
    hit = tot = 0
    for t in np.where(run["dec"])[0]:
        for l in range(L):
            row = run["ids"][t, l]
            hit += sum(1 for x in row if int(x) in seen[l]); tot += K
    return {"prefill_hit": ph, "decode_seen_in_prefill": hit / max(1, tot),
            "prefill_steps": int(pre.sum())}

def persistent_experts(run):
    """Per layer: experts touched in >=50% of decode steps, and their event share."""
    d = np.where(run["dec"])[0]
    n = len(d)
    fracs = []
    for l in range(L):
        cnt = np.bincount(run["ids"][d, l].ravel(), minlength=E) / max(1, n)
        core = cnt >= 0.5
        fracs.append(float((cnt[core].sum() / K)))  # share of that layer's events
    return {"core_count_avg": float(np.mean([(np.bincount(run["ids"][d,l].ravel(),minlength=E) >= 0.5*n).sum() for l in range(L)])),
            "core_event_share_avg": float(np.mean(fracs))}

def heat_per_layer(path):
    counts = np.zeros((L, E))
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("//") or not line.startswith("{"): continue
        parts = line.strip("{},").split(",")
        if len(parts) < 3: continue
        l, e, c = int(parts[0]), int(parts[1]), int(parts[2])
        if 0 <= l < L and 0 <= e < E: counts[l, e] = c
    top16, top64, uniq = [], [], []
    for l in range(L):
        row = np.sort(counts[l])[::-1]
        tot = row.sum()
        top16.append(row[:16].sum() / tot if tot else 0)
        top64.append(row[:64].sum() / tot if tot else 0)
        uniq.append(int((counts[l] > 0).sum()))
    return dict(top16=float(np.mean(top16)), top64=float(np.mean(top64)), unique=float(np.mean(uniq)))

def main():
    base = os.path.join(REPO, "bench", "results", "hitrate")
    outdir = r"C:\Users\Asus\code\deepmoe\build"
    all_sum = {}
    for name in ("base_auto", "base_88g", "stall_off"):
        d = os.path.join(base, name)
        if not os.path.exists(os.path.join(d, "route.bin")):
            continue
        s = summarize(name, d)
        all_sum[name] = s
        print(f"\n===== {name} (cache {s['server']} slots) =====")
        print(f"steps={s['steps']} decode={s['decode_steps']} overall_hit={s['overall_hit']:.4f}")
        for g, v in s["groups"].items():
            print(f"  {g:14s} hit={v['hit']:.4f} top16_share={v['top16']:.3f} unique/layer={v['unique']:.1f}")
        print(f"  prefill->decode: {s['prefill_reuse']}")
        print(f"  persistent core: {s['persistent']}")
        print(f"  reuse within previous k steps: {s['locality']}")
        print(f"  turn stability: {s['turn_stability']}")
        print(f"  hit by position bin: {s['position_bins']}")
        print("  layer  hit    top16  uniq | layer  hit    top16  uniq")
        for l in range(0, 40, 2):
            a, b = s["per_layer"][l], s["per_layer"][l + 1]
            print(f"   {l:2d}  {a['hit']:.3f}  {a['top16']:.3f}  {a['unique']:3d} |"
                  f"   {l+1:2d}  {b['hit']:.3f}  {b['top16']:.3f}  {b['unique']:3d}")
    with open(os.path.join(outdir, "expert_pattern_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_sum, f, ensure_ascii=False, indent=1)
    # capacity sweep summary
    p = os.path.join(base, "base_auto", "capacity_sweep.json")
    if os.path.exists(p):
        cs = json.load(open(p, encoding="utf-8"))["total"]
        print("\n===== base_auto capacity sweep (simulated, same route) =====")
        for k in sorted(k for k in cs if k.startswith("sim_")):
            print(f"  {k}: {cs[k]:.4f}")
        for k in sorted(k for k in cs if k.startswith("heat_")):
            print(f"  {k}: {cs[k]:.4f}")
    # static heat concentration from the 27k mixed trace
    hp = os.path.join(REPO, "store", "static_heat.inc")
    if os.path.exists(hp):
        hc, rows = heat_stats(hp)
        flat = np.sort(hc.ravel())[::-1]
        tot = flat.sum()
        print(f"\n===== store/static_heat.inc: {rows} entries, total {tot} =====")
        for k in (16, 64, 256, 1024, 4096, 15360):
            print(f"  top {k:5d} experts' share: {flat[:k].sum() / tot:.3f}")
        print(f"  per-layer static heat: {heat_per_layer(hp)}")

if __name__ == "__main__":
    main()
