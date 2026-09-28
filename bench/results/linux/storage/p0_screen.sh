#!/bin/bash
# P0 read shape screen: chunk size / QD / in-flight bytes. Internal disk only, serial.
cd ~/projects/cachedMoE
D=bench/results/linux/storage
for arm in "$@"; do
  # arm = cCHUNKqQDmMB, e.g. c2q48m160; "base" = defaults
  env=()
  if [ "$arm" != base ]; then
    c=$(echo $arm | sed -E 's/c([0-9]+)q.*/\1/'); q=$(echo $arm | sed -E 's/.*q([0-9]+)m.*/\1/'); m=$(echo $arm | sed -E 's/.*m([0-9]+)/\1/')
    env=(--env DEEPMOE_IO_P0_CHUNK_MB=$c --env DEEPMOE_IO_P0_QD=$q --env DEEPMOE_IO_P0_INFLIGHT_MB=$m)
  fi
  i=$(ls -d $D/[0-9]*_* 2>/dev/null | wc -l); i=$((i+1)); out=$D/${i}_$arm
  echo "== $i $arm $(date +%T)"
  .venv/bin/python tools/hitrate_bench.py --script bench/results/hitrate/long_turns.json --out $out "${env[@]}" > $out.log 2>&1
  echo "rc=$? $(date +%T)"
done
