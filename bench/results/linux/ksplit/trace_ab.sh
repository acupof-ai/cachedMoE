#!/bin/bash
# Hot-step trace, attention K-split (wo_a/wo_b) off vs on, ABAB.
cd ~/projects/cachedMoE
D=bench/results/linux/ksplit
for arm in off on off on; do
  i=$((i+1)); ks=0; [ $arm = on ] && ks=1
  DEEPMOE_ATTN_KSPLIT=$ks ./build/deepmoe run --model $HOME/models/DeepSeek-V4.1-Flash \
    --steps 8 --warm 8 --slow-prefill --trace $D/t${i}_$arm.bin > $D/t${i}_$arm.txt 2>&1
  echo "== $i $arm rc=$?"
  .venv/bin/python tools/trace_timeline.py $D/t${i}_$arm.bin --summary 2>&1 | grep -E 'GPU span|attention '
done
