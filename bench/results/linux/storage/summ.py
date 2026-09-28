import json, glob, os, re, sys
base = os.path.dirname(os.path.abspath(__file__))
dirs = sorted(glob.glob(base + "/[0-9]*_*/"), key=lambda d: int(os.path.basename(d[:-1]).split("_")[0]))
dirs += [base + "/../pfmin/4_newdefault/"]
for d in dirs:
    try:
        t = json.load(open(d + "turns.json"))
    except Exception as e:
        print(os.path.basename(d[:-1]), "no turns"); continue
    T = [x for x in t["turns"] if x.get("event") == "done"]
    g = sum(x["decode_steps"] for x in T); ms = sum(x["decode_ms"] for x in T)
    n = sum(x["sampled_steps"] for x in T)
    st = sum(x["per_token_ms"]["nvme_stall"] * x["sampled_steps"] for x in T) / n
    hit = sum(x["decode_hit_rate"] * x["decode_steps"] for x in T) / g
    io = t["status"].get("io", "") if isinstance(t["status"], dict) else ""
    eff = re.search(r"eff ([\d.]+) GB/s", io)
    p0 = re.search(r"P0: \d+ req, [\d.]+ MiB, lat mean ([\d.]+)", io)
    fb = re.search(r"first-of-burst \d+ at ([\d.]+) ms, behind \d+ at ([\d.]+) ms", io)
    wall = sum(x["total_ms"] for x in T) / 1000
    print(f"{os.path.basename(os.path.normpath(d)):16s} decode {g/ms*1000:.3f}  e2e {sum(x['generated'] for x in T)/wall:.3f}  hit {hit:.4f}  "
          f"stall {st:5.1f} ms  eff {eff.group(1) if eff else '-'}  P0 lat {p0.group(1) if p0 else '-'}  "
          f"first/behind {fb.group(1)+'/'+fb.group(2) if fb else '-'}")
