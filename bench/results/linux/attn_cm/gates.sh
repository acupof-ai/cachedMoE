#!/bin/bash
# decode_attn_cm (DEEPMOE_ATTN_CM=1) through the numerics gates, one GPU job at
# a time: the two drift suites it did not finish before a reboot, then the
# 64-step l3_ppl for the default and for coopmat attention.
cd ~/projects/cachedMoE
D=bench/results/linux/attn_cm
export DEEPMOE_MODEL_DIR=$HOME/models/DeepSeek-V4.1-Flash DEEPMOE_LONGCTX_DIR=$PWD/traces/longctx
for t in spec_forward. decode_longctx.; do
  DEEPMOE_ATTN_CM=1 ./build/tests/deepmoe_tests $t > $D/cm_${t%.}.log 2>&1
  echo "== $t rc=$?"
  grep -E "worst|ASSERTED|top-1|FAIL|case\(s\)" $D/cm_${t%.}.log | tail -8
done
for arm in 0 1; do
  DEEPMOE_ATTN_CM=$arm .venv/bin/python tools/l3_ppl.py --modes off --json $D/l3ppl_cm$arm.json \
    --log-dir $D/l3ppl_cm${arm}_logs > $D/l3ppl_cm$arm.txt 2>&1
  echo "== l3_ppl cm=$arm rc=$?"
  grep -iE "nll|top-1|off " $D/l3ppl_cm$arm.txt | tail -4
done
