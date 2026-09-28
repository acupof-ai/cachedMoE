#!/bin/bash
# suite.decode_longctx under the same four fp32 summation orders (K-split is the RADV default from here on).
cd ~/projects/cachedMoE
export DEEPMOE_MODEL_DIR=~/models/DeepSeek-V4.1-Flash
out=bench/results/linux/drift_noise
declare -A V=(
  [lc_base]="DEEPMOE_ATTN_KSPLIT=0"
  [lc_oldB]="DEEPMOE_ATTN_KSPLIT=0 DEEPMOE_MOE_LB=32 DEEPMOE_MOE_RB=1"
  [lc_ksplit]=""
  [lc_ksplit_oldB]="DEEPMOE_MOE_LB=32 DEEPMOE_MOE_RB=1"
  [lc_ksplit_repeat]=""
)
for k in lc_base lc_oldB lc_ksplit lc_ksplit_oldB lc_ksplit_repeat; do
  env ${V[$k]} ctest --test-dir build -j 1 -R '^suite\.decode_longctx$' -V > $out/$k.log 2>&1
  echo "rc=$?" >> $out/$k.log
done
.venv/bin/python - <<'PY'
import re, glob, os
for k in ["lc_base", "lc_oldB", "lc_ksplit", "lc_ksplit_oldB", "lc_ksplit_repeat"]:
    t = open(f"bench/results/linux/drift_noise/{k}.log").read()
    mins = {}
    for line in re.findall(r"ASSERTED(.*)", t):
        for nm, v in re.findall(r"(\w+)=([\d.]+)", line):
            mins[nm] = min(mins.get(nm, 1.0), float(v))
    print(f"{k:18s}", " ".join(f"{n} {v:.5f}" for n, v in mins.items()), re.findall(r"rc=(\d+)", t)[-1])
PY
