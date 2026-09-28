#!/bin/bash
# After max_rows_per_submit (2048): the 4K run must print exactly what the
# one-submit run did (ctx4k_replay128.txt), and 17K must no longer lose the
# device (ctx16k_replay128.txt: layer 0, VK_ERROR_DEVICE_LOST).
cd ~/projects/cachedMoE
D=bench/results/linux/replay128
export DEEPMOE_MODEL_DIR=$HOME/models/DeepSeek-V4.1-Flash
for ctx in ctx4k ctx16k; do
  DEEPMOE_PF_LONGCTX=traces/longctx/$ctx DEEPMOE_PF_REPLAY=128 DEEPMOE_PF_DECODE=free \
    ./build/tests/deepmoe_tests gpu_prefill.longctx > $D/${ctx}_replay128_chunked.txt 2>&1
  echo "== $ctx replay128 chunked rc=$?"
  grep -E "first token|worst:|free-running|FAIL|case\(s\)" $D/${ctx}_replay128_chunked.txt | tail -6
done
# the numbers both 4K runs print, minus wall-clock times
strip() { grep -vE "^ +[0-9.]+ s: |kernels:|wall|ms" "$1"; }
if diff <(strip $D/ctx4k_replay128.txt) <(strip $D/ctx4k_replay128_chunked.txt) > $D/ctx4k_diff.txt; then
  echo "4K: identical to the one-submit run"; else echo "4K: DIFFERS ($(wc -l < $D/ctx4k_diff.txt) diff lines)"; fi
