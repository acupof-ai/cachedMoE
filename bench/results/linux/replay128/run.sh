#!/bin/bash
# Production-mode GPU prefill (replay 128, the chat default for prompts > 128
# tokens) of the 4K long-context export, then eight free-running decode steps
# from OUR state -- p3_longctx_decode.md §4.3 ran this only in oracle mode.
cd ~/projects/cachedMoE
D=bench/results/linux/replay128
export DEEPMOE_MODEL_DIR=$HOME/models/DeepSeek-V4.1-Flash
for ctx in ${CTXS:-ctx4k}; do
  DEEPMOE_PF_LONGCTX=traces/longctx/$ctx DEEPMOE_PF_REPLAY=128 DEEPMOE_PF_DECODE=free \
    ./build/tests/deepmoe_tests gpu_prefill.longctx > $D/${ctx}_replay128.txt 2>&1
  echo "== $ctx replay128 rc=$?"
  grep -E "first token|margin|free|salt|kestrel|window|FAIL|case\(s\)" $D/${ctx}_replay128.txt | tail -14
done
