#!/bin/bash
# Launch the cachedMoE web UI on whatever checkpoint sits at the standard paths
# (~/models/DeepSeek-V4.1-Flash and its /mnt/deepmoe2 mirror). Refreshing the
# weights means replacing those directories; this script and the systemd unit
# never name a version.
# Baseline: dynamic 5500-slot mask, k2/top4, CPU routing, native BF16, two read
# sources, 1M context, 4 GB disk KV.
set -eu
cd /home/chenkailun/projects/cachedMoE

export CACHEDMOE_DSPARK_HEAD_FP8=0
export CACHEDMOE_DSPARK_ONECB=1
export CACHEDMOE_BATCH_GPU_ROUTE=0
export CACHEDMOE_DSPARK_MEGA=0
export CACHEDMOE_DSPARK_PROFILE=0
export CACHEDMOE_DSPARK_TRIM_TAIL=1
export CACHEDMOE_SPEC_GPU_READOUT=1
export CACHEDMOE_MGT_PAIR_DOT=0
export CACHEDMOE_MGT_ATTN_CM=0
export CACHEDMOE_MGT_FOLD_SCALE=0
export CACHEDMOE_MASK_DYNAMIC_LRU=1
export CACHEDMOE_MIRROR_AUTO=0
export CACHEDMOE_MODEL_DIR=/home/chenkailun/models/DeepSeek-V4.1-Flash
export CACHEDMOE_SHADER_DIR=/home/chenkailun/projects/cachedMoE/build/shaders
export PYTHONUNBUFFERED=1

exec .venv/bin/python tools/web/server.py \
  --exe build/cachedmoe \
  --resident-only mask \
  --mask-cache dynamic \
  --cache-slots 5500 \
  --max-context 1048576 \
  --gpu-prefill-min 16 \
  --mirror /mnt/deepmoe2/models/DeepSeek-V4.1-Flash \
  --kv-dir build/web_mask/kv \
  --kv-max-gb 8 \
  --dspark --spec-k 2 --spec-top-k 4 \
  --api-audit-dir /home/chenkailun/cmoeapi/api-audit --host 0.0.0.0 --port 8080 \
  --log build/web_mask/engine.log \
  --thermal-policy-json '{"gpu_pause_c":85.0,"gpu_resume_c":77.0,"gpu_start_c":60.0,"nvme_pause_c":82.0,"nvme_resume_c":78.0,"nvme_start_c":76.0}'
