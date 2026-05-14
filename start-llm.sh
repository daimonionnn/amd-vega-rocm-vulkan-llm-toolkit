#!/bin/bash
# Quick-start: Llama 2 7B Chat — Vulkan on Vega8 by default
#
# Kills anything on port 8081, then launches llama-server.
# API: http://127.0.0.1:8081/v1
#
# Defaults to Vulkan backend on the RTX 5090 (Vulkan1).
# ROCm/HIP compute is broken on kernel 6.17 (hard-crashes the PC).
# Vulkan works perfectly via Mesa RADV / NVIDIA proprietary drivers.
#
# Usage:
#   ./start-llm.sh                # Vulkan on iGPU
#   ./start-llm.sh --vega         # Vulkan on Vega 8 iGPU (~49 t/s prompt, ~14 t/s gen)
#   ./start-llm.sh --cpu          # CPU-only via ROCm build (~55 t/s prompt, ~12 t/s gen)
#   ./start-llm.sh --rocm         # ROCm GPU offload (WARNING: crashes on kernel 6.17)
#
# Backends:
#   Vulkan (default) — uses Mesa RADV or NVIDIA proprietary driver
#   ROCm (legacy)    — HIP 5.7 compute, broken on kernel 6.17

set -euo pipefail

export HSA_OVERRIDE_GFX_VERSION=9.0.0
export HSA_ENABLE_SDMA=0
export HCC_SERIALIZE_KERNEL=3
export HCC_SERIALIZE_COPY=3

#MODEL="/home/matt/.lmstudio/models/TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_S.gguf"
MODEL="/home/matt/.lmstudio/models/lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf"
PORT=8080

# ─── Defaults: Vulkan0 ───
BACKEND="vulkan"
VULKAN_DEV="Vulkan0"     # Vega8
NGL=99
CTX=50000
EXTRA_ARGS=()

# ─── Minimum available RAM (MB) required to proceed ───
# LLM model + runtime overhead
MIN_AVAIL_MB=20000

case "${1:-}" in
    --vega)
        VULKAN_DEV="Vulkan0"   # AMD Radeon Vega 8 (RADV RENOIR)
        NGL=99
        CTX=50000
        echo "  Mode: Vulkan on Vega 8 iGPU"
        shift
        ;;
    --cpu)
        BACKEND="rocm"
        NGL=0
        CTX=50000
        export HIP_VISIBLE_DEVICES=-1   # hide GPU, pure CPU mode
        echo "  Mode: CPU-only"
        shift
        ;;
    --rocm)
        BACKEND="rocm"
        NGL=1
        CTX=512
        echo "⚠  ROCm GPU mode — may hard-crash the PC on kernel 6.17!"
        echo "   (HIP compute ring timeouts + MODE2 GPU reset)"
        echo ""
        shift
        ;;
    --help|-h)
        echo "Usage: $0 [--vega|--cpu|--rocm|--help]"
        echo ""
        echo "  (default)   Vulkan on RTX"
        echo "  --vega      Vulkan on Vega 8 iGPU"
        echo "  --cpu       CPU-only (ROCm build)"
        echo "  --rocm      ROCm GPU offload (CRASHES on kernel 6.17)"
        echo ""
        exit 0
        ;;
    *)
        echo "  Mode: Vulkan on RTX "
        ;;
esac


# Kill existing process on port 8080
PID=$(lsof -ti :$PORT 2>/dev/null) || true
if [ -n "$PID" ]; then
    echo "Killing process $PID on port $PORT..."
    kill -9 "$PID" 2>/dev/null || true
    sleep 1
fi

exec "$(dirname "$0")/run/run-llamaserver-${BACKEND}.sh" \
    "$MODEL" \
    -ngl "$NGL" -c "$CTX" --port "$PORT" \
    -b 64 -ub 64 \
    --no-warmup \
    ${BACKEND:+$([ "$BACKEND" = "vulkan" ] && echo "-dev $VULKAN_DEV" || echo "-fa off")} \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
