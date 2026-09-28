#!/bin/bash
# RADV auto budget from the GTT heaps (no 5,000 cap) vs the old cap. Internal disk only, serial ABAB.
cd ~/projects/cachedMoE
D=bench/results/linux/reboot112/auto_ab
i=0
for arm in cap5000 auto cap5000 auto; do
  i=$((i+1)); out=$D/${i}_$arm
  env=()
  [ $arm = cap5000 ] && env=(--env DEEPMOE_CACHE_SLOT_CAP=5000)
  echo "== $i $arm $(date +%T)"
  ( while sleep 20; do cat /sys/class/drm/card1/device/mem_info_gtt_used; done ) > $out.gtt & W=$!
  .venv/bin/python tools/hitrate_bench.py --script bench/results/hitrate/long_turns.json --out $out "${env[@]}" > $out.log 2>&1
  echo "rc=$? $(date +%T) gtt_peak=$(( $(sort -n $out.gtt | tail -1) / 1073741824 )) GiB"
  kill $W
  grep -hE "slot cap|GTT heaps" $out/serve.log | head -2
done
