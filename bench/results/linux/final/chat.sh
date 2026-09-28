#!/bin/bash
# One full 8-turn chat on the defaults after this round (DecodeMode 3, K-split,
# gate.topk / wkv.finish rewrites, P0 1 MiB x 8), then the same with coopmat
# attention on. Single internal disk (hitrate_bench pins DEEPMOE_MIRROR_AUTO=0).
cd ~/projects/cachedMoE
F=bench/results/linux/final
for arm in default attncm; do
  O=$F/chat_$arm; mkdir -p $O
  extra=(); [ $arm = attncm ] && extra=(--env DEEPMOE_ATTN_CM=1)
  .venv/bin/python tools/hitrate_bench.py --script bench/results/hitrate/long_turns.json \
      --out $O "${extra[@]}" > $O.log 2>&1
  echo "== $arm rc=$?"
  .venv/bin/python tools/perf_report.py --chat $O --record "final-$arm-8turns" | sed -n 2,5p
done
