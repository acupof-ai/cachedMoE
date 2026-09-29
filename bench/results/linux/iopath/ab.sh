#!/bin/bash
# Codex review of c92d5ea: isolate the submit-thread change on ONE binary, the
# 8-turn chat with only DEEPMOE_IO_SUBMIT_THREADS switched (8 = the old default,
# 1 = the new). Single internal disk. A failed bench stops before anything is
# recorded, and an existing output directory is refused rather than reused.
set -e
cd ~/projects/cachedMoE
D=bench/results/linux/iopath
for t in 8 1; do
  out=$D/ab_threads$t
  mkdir $out
  .venv/bin/python tools/hitrate_bench.py --script bench/results/hitrate/long_turns.json --out $out \
    --env DEEPMOE_IO_SUBMIT_THREADS=$t > $out.log 2>&1
  .venv/bin/python tools/perf_report.py --chat $out --record chat-ab-threads$t-8turns > ${out}_report.txt 2>&1
  echo "== threads $t"; sed -n 2,4p ${out}_report.txt
done
