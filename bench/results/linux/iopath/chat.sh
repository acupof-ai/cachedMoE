#!/bin/bash
# The 8-turn chat with IoEngine submitting on the dispatcher (Linux default
# submit threads 8 -> 1), against ledger #20 (chat-waves-8turns, 8 threads).
# Single internal disk (hitrate_bench pins DEEPMOE_MIRROR_AUTO=0).
set -e
cd ~/projects/cachedMoE
D=bench/results/linux/iopath
.venv/bin/python tools/hitrate_bench.py --script bench/results/hitrate/long_turns.json --out $D/chat_submit1 > $D/chat_submit1.log 2>&1
.venv/bin/python tools/perf_report.py --chat $D/chat_submit1 --record chat-submit1-8turns > $D/chat_submit1_report.txt 2>&1
cat $D/chat_submit1_report.txt
