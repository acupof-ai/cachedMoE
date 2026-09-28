#!/bin/bash
# P0 chunk / QD neighbourhood, 3 turns each (stall-per-miss is stable to ~1% there).
cd ~/projects/cachedMoE
for cfg in "1 4" "1 12" "2 8" "1 8"; do
  set -- $cfg
  O=bench/results/linux/perf/p0sweep/c${1}q${2}
  mkdir -p $O
  .venv/bin/python tools/hitrate_bench.py --script bench/results/hitrate/long_turns.json --max-turns 3 \
      --out $O --env DEEPMOE_IO_P0_CHUNK_MB=$1 --env DEEPMOE_IO_P0_QD=$2 > $O.log 2>&1
  .venv/bin/python tools/perf_report.py --chat $O --record "p0sweep-c${1}q${2}-3turns" | sed -n 2,5p
done
