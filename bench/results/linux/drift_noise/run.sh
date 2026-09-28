#!/bin/bash
# How far do arithmetic-equivalent variants move suite.decode's and
# suite.spec_forward's drift numbers? Each variant changes only fp32 summation
# order (or nothing: hq2 is byte-identical to the default, the control).
cd ~/projects/cachedMoE
export DEEPMOE_MODEL_DIR=~/models/DeepSeek-V4.1-Flash
out=bench/results/linux/drift_noise
declare -A V=(
  [base]=""
  [hq2_control]="DEEPMOE_MOE_HQUANT=2"
  [ksplit]="DEEPMOE_ATTN_KSPLIT=1"
  [oldB]="DEEPMOE_MOE_LB=32 DEEPMOE_MOE_RB=1"
  [se0]="DEEPMOE_SHARED_EARLY=0"
  [ksplit_oldB]="DEEPMOE_ATTN_KSPLIT=1 DEEPMOE_MOE_LB=32 DEEPMOE_MOE_RB=1"
  [base_repeat]=""
)
for k in ${ONLY:-base hq2_control ksplit oldB se0 ksplit_oldB base_repeat}; do
  env ${V[$k]} ctest --test-dir build -j 1 -R '^suite\.(decode|spec_forward)$' -V > $out/$k.log 2>&1
  echo "rc=$?" >> $out/$k.log
done
.venv/bin/python - <<'PY'
import re, glob, os, json
rows = {}
for f in sorted(glob.glob('bench/results/linux/drift_noise/*.log')):
    t = open(f).read(); k = os.path.basename(f)[:-4]
    m = re.search(r'compressed KV ([\d.]+) \(L(\d+).*?index keys ([\d.]+) \(L(\d+)', t)
    s = re.search(r'(\d+) positions: worst cos ([\d.]+) at \d+, max \|dlogit\| ([\d.e+-]+), top-1 (\d+)/(\d+)', t)
    rows[k] = {"cmp": m and float(m.group(1)), "key": m and float(m.group(3)),
               "spec_cos": s and float(s.group(2)), "spec_top1": s and f"{s.group(4)}/{s.group(5)}",
               "rc": re.findall(r'rc=(\d+)', t)[-1]}
    print(f"{k:14s} cmp {rows[k]['cmp']}  key {rows[k]['key']}  spec cos {rows[k]['spec_cos']} top-1 {rows[k]['spec_top1']}  rc {rows[k]['rc']}")
json.dump(rows, open('bench/results/linux/drift_noise/summary.json', 'w'), indent=1)
PY
