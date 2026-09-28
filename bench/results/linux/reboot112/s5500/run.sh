#!/bin/bash
# After reboot: carve-out 512 MB + 112 GiB GTT. args: list of arm names (off|on|onNNNN|offNNNN)
cd ~/projects/cachedMoE
M=/mnt/deepmoe2/models/DeepSeek-V4.1-Flash
i=0
for arm in "$@"; do
  i=$((i+1)); out=bench/results/linux/reboot112/s5500/${i}_$arm
  args=()
  case $arm in on*) args=(--serve-arg=--mirror --serve-arg=$M);; esac
  slots=${arm#on}; slots=${slots#off}
  [ -n "$slots" ] && args+=(--cache-slots $slots)
  echo "== $i $arm $(date +%T)"
  ( while sleep 20; do cat /sys/class/drm/card1/device/mem_info_gtt_used; done ) > $out.gtt & W=$!
  .venv/bin/python tools/hitrate_bench.py --script bench/results/hitrate/long_turns.json --out $out "${args[@]}" > $out.log 2>&1
  echo "rc=$? $(date +%T) gtt_peak=$(( $(sort -n $out.gtt | tail -1) / 1073741824 )) GiB"
  kill $W
  grep -hE "slot cap|dropped|re-read" $out/serve.log | head -3
  findmnt -n /mnt/deepmoe2 >/dev/null || { echo "external disk gone"; break; }
done
