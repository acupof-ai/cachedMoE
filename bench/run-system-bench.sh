#!/bin/sh
# 一条命令跑完 cachedMoE 的系统级基准，输出可直接对比的四个数。
# 覆盖：MoE kernel 带宽 / attention 带宽 / NVMe 读速 / I/O 目的地开销。
# 端到端 tok/s 要加载 checkpoint，几分钟，默认不跑；加 --full 打印跑法。
# 一次只允许一个 GPU job（AGENTS.md），所以这里是串行的。
set -u
cd "$(dirname "$0")/.." || exit 1

export CACHEDMOE_MODEL_DIR="${CACHEDMOE_MODEL_DIR:-$HOME/models/DeepSeek-V4.1-Flash}"
M="$CACHEDMOE_MODEL_DIR"
B=build
[ -x "$B/kernel_bench" ] || { echo "先构建: cmake --build build"; exit 1; }
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "== 1/4 MoE kernel — 目标 >=80% raw read ceiling"
"$B/kernel_bench" --quick 2>&1 | grep -E "best M=1 variant|per-dispatch launch" | sed 's/^/  /'

echo "== 2/4 attention — 目标 >=80% UMA(216 GB/s)"
"$B/attn_bench" --model "$M" --layers 2 --iters 6 2>&1 \
  | grep -E "40 layers \+ head|one layer, P3" | sed 's/^/  /'

echo "== 3/4 NVMe — 目标 >=5.0 GB/s 单点读"
"$B/nvme_bench" --file "$TMP/nv.tmp" --size-gb 2 --chunk-kb 4096,18432 \
  --qd 4,8 --reads 12 --pattern both 2>&1 | grep -E "^best|one 18.8 MB" | sed 's/^/  /'

echo "== 4/4 I/O 目的地内存成本 — stage 相对 patha 应 <5%"
"$B/io_dst_bench" --req-kb 9184 --chunk-kb 4096 --qd 4,8 --dst ram,patha --gpu-copy 2>&1 \
  | awk '/^(ram|patha|stage)/ {print "  " $0}'

if [ "${1:-}" = "--full" ]; then
  cat <<'EOF'
== 5/5 端到端（手动，加载 checkpoint）
  启动: tools/web/run-web.sh  或  build/cachedmoe serve
  计时: 单轮 64 token，记 tok/s 与 hit
  判定: tok/s ≈ 8.3–9.2 GB/s ÷ MB_per_token（STATUS §4）
EOF
fi

echo
echo "判定基准（STATUS §1/§4）：MoE 92% / attention 95% / NVMe 8.9 GB/s 是这台机器当前水平。"
