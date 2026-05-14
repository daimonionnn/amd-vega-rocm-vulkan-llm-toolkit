#!/bin/bash
# start-llama-server.sh — Launch llama-server on Vega 8 iGPU
#
# Defaults to ROCm 7.2 baremetal (best GPU prefill, no Docker needed).
# Model defaults to Qwen3.5-35B-A3B-Q4_K_M — override with MODEL= env var.
#
# Usage:
#   ./run/start-llama-server.sh                # ROCm 7.2 baremetal (default)
#   ./run/start-llama-server.sh --vulkan       # Vulkan / Mesa RADV
#   ./run/start-llama-server.sh --cpu          # CPU only (no GPU offload)
#   ./run/start-llama-server.sh --rocm-docker  # ROCm 7.2 via Docker
#
# Environment variables:
#   MODEL=/path/to/model.gguf        — override default model path
#   CTX=8192                         — context size (default: 8192)
#   PORT=8080                        — server port  (default: 8080)
#
# Backends at a glance (Vega 8 iGPU, Qwen3.5-35B-A3B-Q4_K_M):
#   ROCm 7.2 baremetal  — prefill ~68 t/s @ 4K ctx, decode ~14 t/s
#   Vulkan (Mesa RADV)  — prefill ~50 t/s @ 4K ctx, decode ~20 t/s  ← best decode
#   CPU only            — prefill ~233 t/s @ 4K ctx, decode ~13 t/s ← best prefill
#
# See docs/benchmarks.md for full comparison.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL="${MODEL:-$HOME/.lmstudio/models/lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf}"
PORT="${PORT:-8080}"
CTX="${CTX:-8192}"

# ─── Parse mode ──────────────────────────────────────────────────────────────

MODE="${1:-}"

case "$MODE" in
    --vulkan)
        shift
        echo "  Backend: Vulkan (Mesa RADV / Vega 8)"
        echo "  API:     http://127.0.0.1:$PORT/v1"
        echo ""
        PID=$(lsof -ti :"$PORT" 2>/dev/null) || true
        [ -n "$PID" ] && { echo "Killing existing process $PID on :$PORT"; kill -9 "$PID" 2>/dev/null || true; sleep 1; }
        exec "$SCRIPT_DIR/run-llamaserver-vulkan.sh" \
            "$MODEL" \
            -ngl 99 -c "$CTX" --port "$PORT" --no-warmup \
            -dev Vulkan0 -fa 1 \
            "$@"
        ;;
    --cpu)
        shift
        echo "  Backend: CPU only (no GPU offload)"
        echo "  API:     http://127.0.0.1:$PORT/v1"
        echo ""
        PID=$(lsof -ti :"$PORT" 2>/dev/null) || true
        [ -n "$PID" ] && { echo "Killing existing process $PID on :$PORT"; kill -9 "$PID" 2>/dev/null || true; sleep 1; }
        exec "$SCRIPT_DIR/run-llamaserver-vulkan.sh" \
            "$MODEL" \
            -ngl 0 -c "$CTX" --port "$PORT" --no-warmup \
            -fa 1 \
            "$@"
        ;;
    --rocm-docker)
        shift
        echo "  Backend: ROCm 7.2 Docker"
        echo "  API:     http://127.0.0.1:$PORT/v1"
        echo ""
        PID=$(lsof -ti :"$PORT" 2>/dev/null) || true
        [ -n "$PID" ] && { echo "Killing existing process $PID on :$PORT"; kill -9 "$PID" 2>/dev/null || true; sleep 1; }
        exec "$SCRIPT_DIR/run-docker-rocm7.sh" \
            "$MODEL" \
            -ngl 99 -c "$CTX" --port "$PORT" --no-warmup \
            -fa 0 \
            "$@"
        ;;
    --help|-h)
        echo "Usage: $0 [--vulkan|--cpu|--rocm-docker|--help]"
        echo ""
        echo "  (default)       ROCm 7.2 baremetal — best GPU prefill, no Docker"
        echo "  --vulkan        Vulkan / Mesa RADV  — best decode throughput"
        echo "  --cpu           CPU only            — best prefill at large context"
        echo "  --rocm-docker   ROCm 7.2 in Docker  — same perf as baremetal"
        echo ""
        echo "Env vars: MODEL=  CTX=  PORT="
        echo ""
        exit 0
        ;;
    ""|--rocm|--rocm7|--baremetal)
        [ -n "$MODE" ] && shift || true
        echo "  Backend: ROCm 7.2 baremetal (Vega 8, gfx900)"
        echo "  API:     http://127.0.0.1:$PORT/v1"
        echo ""
        PID=$(lsof -ti :"$PORT" 2>/dev/null) || true
        [ -n "$PID" ] && { echo "Killing existing process $PID on :$PORT"; kill -9 "$PID" 2>/dev/null || true; sleep 1; }
        exec "$SCRIPT_DIR/run-rocm7-baremetal.sh" \
            "$MODEL" \
            -ngl 99 -c "$CTX" --port "$PORT" --no-warmup \
            -fa 0 \
            "$@"
        ;;
    *)
        echo "Unknown option: $MODE"
        echo "Run '$0 --help' for usage."
        exit 1
        ;;
esac
