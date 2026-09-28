#!/bin/bash
cd ~/projects/cachedMoE
P=bench/results/linux/perf
.venv/bin/python tools/hitrate_bench.py --script bench/results/hitrate/long_turns.json --out $P/gm_single > $P/gm_single.log 2>&1
echo "single rc=$?"; grep -h gamemode $P/gm_single/serve.log | head -1
.venv/bin/python tools/hitrate_bench.py --script bench/results/hitrate/long_turns.json --out $P/gm_stripe --env DEEPMOE_MIRROR_AUTO=1 --env DEEPMOE_MIRROR_STRIPE=1 > $P/gm_stripe.log 2>&1
echo "stripe rc=$?"
cat /sys/class/drm/card1/device/power_dpm_force_performance_level
