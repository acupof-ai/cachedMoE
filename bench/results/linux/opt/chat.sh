#!/bin/bash
# The batch after hot-engram-async (projection waves, mega_mhc Sinkhorn split),
# measured once: the hot step, then the 8-turn chat on the defaults. Single
# internal disk (hitrate_bench pins DEEPMOE_MIRROR_AUTO=0).
cd ~/projects/cachedMoE
O=bench/results/linux/opt
.venv/bin/python tools/perf_report.py --capture --record hot-waves --json $O/hot_waves.json > $O/hot_waves.txt 2>&1
mkdir -p $O/chat_waves
.venv/bin/python tools/hitrate_bench.py --script bench/results/hitrate/long_turns.json --out $O/chat_waves > $O/chat_waves.log 2>&1
echo "== rc=$?"
.venv/bin/python tools/perf_report.py --chat $O/chat_waves --record chat-waves-8turns > $O/chat_waves_report.txt 2>&1
