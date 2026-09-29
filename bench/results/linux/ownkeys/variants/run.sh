#!/bin/bash
# suite.decode_longctx against the own-keys oracles under fp32 summation orders
# that change nothing else: how far does each step's worst cosine move on its own?
cd ~/projects/cachedMoE
export DEEPMOE_MODEL_DIR=~/models/DeepSeek-V4.1-Flash
out=bench/results/linux/ownkeys/variants
declare -A V=(
  [ksplit]=""
  [noksplit]="DEEPMOE_ATTN_KSPLIT=0"
  [oldB]="DEEPMOE_MOE_LB=32 DEEPMOE_MOE_RB=1"
  [noksplit_oldB]="DEEPMOE_ATTN_KSPLIT=0 DEEPMOE_MOE_LB=32 DEEPMOE_MOE_RB=1"
)
for k in ksplit noksplit oldB noksplit_oldB; do
  env ${V[$k]} ctest --test-dir build -j 1 -R '^suite\.decode_longctx$' -V > $out/$k.log 2>&1
  echo "rc=$?" >> $out/$k.log
done
.venv/bin/python - <<'PY'
import re
out = "bench/results/linux/ownkeys/variants"
for k in ["ksplit", "noksplit", "oldB", "noksplit_oldB"]:
    t = open(f"{out}/{k}.log").read()
    w = [float(x) for x in re.findall(r"worst over the probe layers: attn_norm ([\d.]+)", t)]
    print(f"{k:14s} rc {re.findall(r'rc=(\d+)', t)[-1]}  attn_norm per step:", " ".join(f"{x:.3f}" for x in w))
PY
