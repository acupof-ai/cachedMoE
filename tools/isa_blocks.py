#!/usr/bin/env python3
"""Static instruction mix per basic block of a DEEPMOE_PIPELINE_STATS ISA dump.

The hardware instruction counters on this machine under-report (STATUS §7 0bq), so
"how many instructions does this kernel issue" is answered here instead: static
counts per block, times the trip count the caller supplies. A VOPD instruction
(`v_dual_*`) is ONE encoding carrying TWO operations; both are reported, because
which one matters depends on whether you are counting issue slots or work.

  tools/isa_blocks.py DUMP [--trips BB=N,BB=N,...] [--per-wave]

Without --trips every block counts once. With it, the named blocks are weighted and
the total is the dynamic per-wave instruction count.
"""
import re, sys, collections

def blocks(path):
    txt = open(path).read()
    # the final assembly is the last section; blocks are lines starting BBn
    tail = txt.split("Assembly (Final Assembly)")[-1]
    out, cur, name = {}, [], None
    for line in tail.splitlines():
        m = re.match(r"^BB(\d+):?\s*$", line.strip())
        if m:
            if name is not None: out[name] = cur
            name, cur = f"BB{m.group(1)}", []
        elif name is not None:
            cur.append(line)
    if name is not None: out[name] = cur
    return out

CLASSES = [
    ("wmma",    lambda l: "v_wmma" in l),
    ("vopd",    lambda l: "v_dual_" in l),
    ("valu",    lambda l: re.search(r"=\s*v_", l) and "v_dual_" not in l and "v_wmma" not in l),
    ("salu",    lambda l: re.search(r"=\s*s_", l) and "s_waitcnt" not in l),
    ("gload",   lambda l: "global_load" in l),
    ("dsread",  lambda l: "ds_read" in l),
    ("dswrite", lambda l: "ds_write" in l),
    ("waitcnt", lambda l: "s_waitcnt" in l),
    ("barrier", lambda l: "s_barrier" in l),
]

def mix(lines):
    c = collections.Counter()
    for l in lines:
        # RADV versions differ: ACO IR uses "%n = v_op", while final ISA
        # starts directly with "v_op ..." and labels its blocks "BBn:".
        final = re.match(r"^\s*([a-z][a-z0-9_]*)\s", l)
        if final:
            op = final.group(1)
            if op.startswith("v_wmma"): c["wmma"] += 1
            elif op.startswith("v_dual_"): c["vopd"] += 1
            elif op.startswith("v_"): c["valu"] += 1
            elif op.startswith("global_load"): c["gload"] += 1
            elif op.startswith("ds_read"): c["dsread"] += 1
            elif op.startswith("ds_write"): c["dswrite"] += 1
            elif op == "s_waitcnt": c["waitcnt"] += 1
            elif op == "s_barrier": c["barrier"] += 1
            elif op.startswith("s_"): c["salu"] += 1
            continue
        for name, pred in CLASSES:
            if pred(l): c[name] += 1
    return c

def main():
    path = sys.argv[1]
    trips = {}
    for a in sys.argv[2:]:
        if a.startswith("--trips="):
            for kv in a.split("=", 1)[1].split(","):
                k, v = kv.split("="); trips[k.strip()] = int(v)
    bs = blocks(path)
    tot = collections.Counter()
    hdr = ["block", "trips"] + [n for n, _ in CLASSES] + ["issue"]
    print("  ".join(f"{h:>8}" for h in hdr))
    for name, lines in bs.items():
        c = mix(lines)
        if not sum(c.values()): continue
        t = trips.get(name, 1)
        # issue slots: every encoding once (a VOPD pair is one)
        issue = c["wmma"] + c["vopd"] + c["valu"] + c["salu"] + c["gload"] + c["dsread"] + c["dswrite"]
        row = [name, str(t)] + [str(c[n]) for n, _ in CLASSES] + [str(issue)]
        print("  ".join(f"{v:>8}" for v in row))
        for k in c: tot[k] += c[k] * t
        tot["issue"] += issue * t
    print("-" * 8 * (len(hdr) + 1))
    row = ["TOTAL", ""] + [str(tot[n]) for n, _ in CLASSES] + [str(tot["issue"])]
    print("  ".join(f"{v:>8}" for v in row))
    if tot["wmma"]:
        print(f"\nwmma {tot['wmma']} of {tot['issue']} issue slots = {tot['wmma']/tot['issue']*100:.1f}%")

main()
