#!/bin/bash
# PR #2 gate step 2: 8-turn chat, both arms with the USB4 mirror, stripe off/on.
cd ~/projects/cachedMoE
i=0
for arm in 0 1 0 1; do
  i=$((i+1)); out=bench/results/linux/stripe/${i}_stripe$arm
  t=$(for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name)" = nvme ] && echo -n "$(basename $(readlink -f $h/device))=$(($(cat $h/temp1_input)/1000))C "; done)
  echo "== $i stripe=$arm $(date +%T) $t"
  .venv/bin/python tools/hitrate_bench.py --script bench/results/hitrate/long_turns.json --out $out \
      --env DEEPMOE_MIRROR_AUTO=1 --env DEEPMOE_MIRROR_STRIPE=$arm > $out.log 2>&1
  echo "rc=$? $(date +%T)"; grep -h "STRIPED\|mirror auto" $out/serve.log | head -2
  findmnt -n /mnt/deepmoe2 >/dev/null || { echo "external disk gone"; break; }
done
