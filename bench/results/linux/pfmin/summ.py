import json, glob, os
base = os.path.dirname(os.path.abspath(__file__))
runs = sorted(glob.glob(base + "/[0-9]_*/")) + [base + "/../reboot112/auto_ab/2_auto/", base + "/../reboot112/auto_ab/4_auto/"]
for d in runs:
    try:
        t = json.load(open(d + "turns.json"))
    except Exception as e:
        print(d, "no turns", e); continue
    T = [x for x in t["turns"] if x.get("event") == "done"]
    g = sum(x["decode_steps"] for x in T); ms = sum(x["decode_ms"] for x in T)
    ttft = [round(x["ttft_ms"] / 1000, 1) for x in T]
    modes = "".join(x["prefill_mode"][0] for x in T)
    wall = sum(x["total_ms"] for x in T) / 1000
    print(f"{os.path.basename(os.path.normpath(d)):10s} decode {g/ms*1000:.3f} tok/s  TTFT sum {sum(ttft):6.1f} s {ttft} modes {modes}  "
          f"gen {sum(x['generated'] for x in T)}  wall {wall:.0f} s  e2e {sum(x['generated'] for x in T)/wall:.3f} tok/s")
