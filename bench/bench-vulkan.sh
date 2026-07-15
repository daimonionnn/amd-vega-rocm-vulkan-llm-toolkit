#!/usr/bin/env bash

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: $0 /path/to/model.gguf [additional llama-bench args]"
    echo "Example: $0 ~/models/llama-2-7b-chat.Q4_K_S.gguf"
    exit 1
fi

MODEL="$1"
shift

BENCH_BIN="$(dirname "$0")/../llm/vulkan/bin/llama-bench"

if [ ! -x "$BENCH_BIN" ]; then
    echo "✗ Error: $BENCH_BIN not found or not executable."
    exit 1
fi

echo "=========================================================="
echo " Starting Vulkan llama-bench"
echo " Model: $MODEL"
echo "=========================================================="

# TODO: no -dev selection — llama-bench picks the default Vulkan device, which
#       on the multi-GPU host may be an R9700 rather than the Vega 8.
#       Auto-detect the RADV RENOIR index (see run/start-llama-server.sh) and
#       pass it here so results are always for the iGPU.
"$BENCH_BIN" -m "$MODEL" -p 128,512,1024,2048 -n 128 -ngl 99 "$@"
