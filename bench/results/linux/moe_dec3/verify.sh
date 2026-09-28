#!/bin/bash
# After DecodeMode 3 became the RADV default: suite.decode on the new default
# (must match the pre-change run), suite.decode with coopmat attention (the log
# the reboot lost), then two hot-step traces for the ledger.
cd ~/projects/cachedMoE
D=bench/results/linux/moe_dec3
export DEEPMOE_MODEL_DIR=$HOME/models/DeepSeek-V4.1-Flash DEEPMOE_LONGCTX_DIR=$PWD/traces/longctx
./build/tests/deepmoe_tests decode. > $D/decode_default.log 2>&1; echo "== decode default rc=$?"
grep -E "teacher-forced|free-running|prefill state vs|case\(s\)" $D/decode_default.log
DEEPMOE_ATTN_CM=1 ./build/tests/deepmoe_tests decode. > bench/results/linux/attn_cm/cm_decode.log 2>&1; echo "== decode cm rc=$?"
grep -E "teacher-forced|free-running|prefill state vs|case\(s\)" bench/results/linux/attn_cm/cm_decode.log
git_head=$(git rev-parse --short HEAD)
.venv/bin/python tools/perf_report.py --capture --record hot-dec3 --json $D/hot_dec3.json > $D/hot_dec3.txt 2>&1; echo "== hot dec3 rc=$?"
head -4 $D/hot_dec3.txt
.venv/bin/python tools/perf_report.py --capture --env DEEPMOE_ATTN_CM=1 --record hot-dec3-attncm --json $D/hot_dec3_cm.json > $D/hot_dec3_cm.txt 2>&1; echo "== hot dec3+cm rc=$?"
head -4 $D/hot_dec3_cm.txt
