#!/bin/bash
#
# Run llama-server (ROCm 7 — baremetal / host install) on Vega 8 iGPU.
#
# Requires ROCm 7.x installed in /opt/rocm and llama.cpp built for gfx900.
# See:
#   setup/install-rocm7-host.sh          — install ROCm 7 on the host
#   build/build-llamacpp-rocm7-baremetal.sh — build llama.cpp (gfx900)
#
# Usage:
#   ./run/run-rocm7-baremetal.sh /path/to/model.gguf [llama-server options]
#   ./run/run-rocm7-baremetal.sh /path/to/model.gguf -ngl 99 -c 8192 -fa 0
#
# Multi-GPU system note:
#   Auto-detects Vega 8 by its rocminfo agent index (gfx90x family).
#   If auto-detect fails, set: VEGA8_ROCM_DEVICE=0 (or the correct index).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_BIN="$REPO_DIR/llm/rocm7-vega/bin/llama-server"
ROCM_LIB_DIR="$REPO_DIR/llm/rocm7-vega/lib"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

# ─── Validate ────────────────────────────────────────────────────────────────

if [ ! -x "$LLAMA_BIN" ]; then
    echo "✗  llama-server not found at: $LLAMA_BIN"
    echo "   Build first: bash build/build-llamacpp-rocm7-baremetal.sh"
    exit 1
fi

if [ -z "${1:-}" ]; then
    echo "Usage: $0 /path/to/model.gguf [llama-server options]"
    echo ""
    echo "  -ngl 99       Offload all layers to GPU"
    echo "  -c 8192       Context size"
    echo "  -fa 0         Flash attention (OFF recommended for ROCm on Vega 8)"
    echo "  --no-warmup   Skip warmup inference"
    echo ""
    echo "Example:"
    echo "  $0 ~/.lmstudio/models/.../model.gguf -ngl 99 -c 8192 -fa 0"
    exit 0
fi

MODEL="$1"
shift

if [ ! -f "$MODEL" ]; then
    echo "✗  Model file not found: $MODEL"
    exit 1
fi

# ─── Detect Vega 8 ROCm agent index ─────────────────────────────────────────
# rocminfo lists agents; Vega 8 shows as gfx90x (typically gfx900 or gfx90c).
# We scan for the first agent matching gfx90 and record its 0-based index.

detect_vega8_rocm_index() {
    if [ -n "${VEGA8_ROCM_DEVICE:-}" ]; then
        echo "$VEGA8_ROCM_DEVICE"
        return
    fi

    local rocminfo_bin="$ROCM_PATH/bin/rocminfo"
    if [ ! -x "$rocminfo_bin" ]; then
        echo "0"   # fallback
        return
    fi

    # Count GPU agents before hitting gfx90x — that gives us the 0-based device index
    local gpu_index=0
    local found_index=""
    while IFS= read -r line; do
        if echo "$line" | grep -q "^Agent [0-9]"; then
            : # new agent block starting
        fi
        if echo "$line" | grep -qi "gfx90"; then
            found_index="$gpu_index"
            break
        fi
        if echo "$line" | grep -qi "Device Type.*GPU"; then
            gpu_index=$((gpu_index))
        fi
        if echo "$line" | grep -q "^Agent [0-9]"; then
            # We reset the per-agent GPU count only when an agent block restarts
            :
        fi
    done < <("$rocminfo_bin" 2>/dev/null)

    # Simpler fallback: just find "gfx90" and count preceding GPU agents
    if [ -z "$found_index" ]; then
        # Count GPU agents that appear before the first gfx90x entry
        found_index=$("$rocminfo_bin" 2>/dev/null | awk '
            /Device Type.*GPU/ { gpu_count++ }
            /gfx90/ { print gpu_count - 1; exit }
        ' || echo "0")
    fi

    echo "${found_index:-0}"
}

VEGA8_IDX=$(detect_vega8_rocm_index)
echo "  Vega 8 ROCm device index: $VEGA8_IDX"

# ─── Environment ─────────────────────────────────────────────────────────────
#
# HSA_OVERRIDE_GFX_VERSION=9.0.0 — Vega 8 APU (gfx90c) overridden to gfx900
#   so that gfx900 tensile kernels are loaded.
# HSA_ENABLE_SDMA=0 — disable System DMA; required for stability on Vega 8 APU
#   (SDMA engine not present / unreliable on integrated Vega).
# GGML_HIP_UMA=1 — Unified Memory Access: model weights accessed via CPU-mapped
#   pointers rather than HIP memcpy; required for iGPU with shared DRAM.
# GPU_MAX_ALLOC_PERCENT=100 — allow full GTT allocation (64 GB on this system).

export ROCR_VISIBLE_DEVICES="$VEGA8_IDX"
export HIP_VISIBLE_DEVICES="$VEGA8_IDX"
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export HSA_ENABLE_SDMA=0
export HSA_XNACK=0
export GGML_HIP_UMA=1
export GPU_MAX_ALLOC_PERCENT=100

# Prepend the baremetal lib dir (contains RPATH-relative libs from build),
# then the ROCm system libs.
export LD_LIBRARY_PATH="${ROCM_LIB_DIR}:${ROCM_PATH}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ─── Launch ───────────────────────────────────────────────────────────────────

echo "  Model:  $(basename "$MODEL")"
echo "  Binary: $LLAMA_BIN"
echo "  ROCm:   $ROCM_PATH"
echo "  Env:    ROCR_VISIBLE_DEVICES=$VEGA8_IDX  HSA_OVERRIDE_GFX_VERSION=9.0.0"
echo ""

exec "$LLAMA_BIN" \
    -m "$MODEL" \
    --host 0.0.0.0 \
    --port 8080 \
    "$@"
