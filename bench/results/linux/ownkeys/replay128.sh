#!/bin/bash
# The chat default (GPU prefill, replay 128) of the 4K / 17K exports, then eight
# free-running decode steps from OUR state, now that every kv source scores its
# own index keys (the p3_longctx_decode.md §4.3 check, re-run on the new rule).
cd ~/projects/cachedMoE
D=bench/results/linux/ownkeys
export DEEPMOE_MODEL_DIR=$HOME/models/DeepSeek-V4.1-Flash
for ctx in ctx4k ctx16k; do
  DEEPMOE_PF_LONGCTX=traces/longctx/$ctx DEEPMOE_PF_REPLAY=128 DEEPMOE_PF_DECODE=free \
    ./build/tests/deepmoe_tests gpu_prefill.longctx > $D/${ctx}_replay128.txt 2>&1
  echo "== $ctx replay128 rc=$?"
  grep -E "first token|free-running|FAIL|case\(s\)" $D/${ctx}_replay128.txt | tail -6
done
