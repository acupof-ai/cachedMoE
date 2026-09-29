#!/bin/bash
# The 8-turn chat with the second read source (the USB4 drive at /mnt/deepmoe2,
# auto-detected) striped per chunk, logging both drives' temperatures once a
# second (hwmon; the external box has dropped off the bus when hot before).
# Auxiliary numbers: the main baseline stays single-disk (ledger #21).
set -e
cd ~/projects/cachedMoE
D=bench/results/linux/multidisk
name=${1:-chat_stripe}
temps() {   # epoch_s int_C ext_C, from the hwmon of whichever nvme holds each mount
  for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name)" = nvme ] || continue
    case "$(readlink -f $h/device)" in *0000:c3:00.0*) i=$h;; *) e=$h;; esac; done
  while :; do echo "$(date +%s.%N | cut -c1-14) $(( $(cat $i/temp1_input) / 1000 )) $(( $(cat $e/temp1_input 2>/dev/null || echo 0) / 1000 ))"; sleep 1; done
}
temps > $D/${name}_temps.txt & TP=$!
trap 'kill $TP 2>/dev/null' EXIT
.venv/bin/python tools/hitrate_bench.py --script bench/results/hitrate/long_turns.json --out $D/$name \
  --env DEEPMOE_MIRROR_AUTO=1 ${STRIPE:+--env DEEPMOE_MIRROR_STRIPE=1} ${HOT_C:+--env DEEPMOE_MIRROR_HOT_C=$HOT_C} > $D/$name.log 2>&1
.venv/bin/python tools/perf_report.py --chat $D/$name --record ${name//_/-}-8turns > $D/${name}_report.txt 2>&1
cat $D/${name}_report.txt
awk '{if($3>m)m=$3; if($2>n)n=$2} END{print "peak temps: internal", n, "C, external", m, "C"}' $D/${name}_temps.txt
grep -iE "mirror|source|DROPPED|drop|resting|cooled" $D/$name/serve.log | head -30
