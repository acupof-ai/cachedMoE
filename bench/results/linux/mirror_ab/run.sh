#!/bin/bash
# Dual-source ABAB: internal SN740 only vs + external USB4 mirror. auto slots.
cd ~/projects/cachedMoE
M=/mnt/deepmoe2/models/DeepSeek-V4.1-Flash
for arm in off on off on; do
  i=$((i+1)); out=bench/results/linux/mirror_ab/${i}_$arm
  args=()
  [ $arm = on ] && args=(--serve-arg=--mirror --serve-arg=$M)
  echo "== $i $arm $(date +%T)"
  .venv/bin/python tools/hitrate_bench.py --script bench/results/hitrate/long_turns.json --out $out "${args[@]}" > $out.log 2>&1
  echo "rc=$? $(date +%T)"
  grep -E "mirror|source '" $out/serve.log | head -8
  findmnt -n /mnt/deepmoe2 >/dev/null || { echo "external disk gone"; break; }
done
