#!/bin/bash
#
# Run llama-server (ROCm 7 / Vega 8 build) inside Docker.
# Mirrors run-docker-rocm.sh but targets the ROCm 7 experimental image.
#
# With multiple AMD GPUs (Vega 8 iGPU + Radeon 9700 AI Pro dGPU) this script
# auto-detects the Vega 8 render node by PCI device ID and passes ONLY that
# device into the container.  ROCR_VISIBLE_DEVICES=0 inside the container is
# therefore always the Vega 8, never the 9700.
#
# Usage:
#   ./run-docker-rocm7.sh /path/to/model.gguf [llama-server options]
#
# Build the image first (one-time, ~20–40 min):
#   docker build -t llama-rocm7-vega -f Dockerfile.rocm7-vega .

set -euo pipefail

IMAGE_NAME="llama-rocm7-vega"
MODEL="${1:-}"

if [ -z "$MODEL" ]; then
    echo "Usage: ./run-docker-rocm7.sh /path/to/model.gguf [llama-server options]"
    exit 1
fi

MODEL_DIR="$(dirname "$MODEL")"
MODEL_NAME="$(basename "$MODEL")"
shift

# ── Detect Vega 8 render node by PCI device ID ───────────────────────────────────────
# Ryzen 5700G Vega 8 = PCI device ID 0x1638  (gfx90c) — confirmed on /dev/dri/renderD129
# Override with VEGA8_RENDER_NODE=/dev/dri/renderDXXX if auto-detect fails.
VEGA8_PCI_ID="0x1638"
VEGA8_RENDER_NODE="${VEGA8_RENDER_NODE:-}"

if [ -z "$VEGA8_RENDER_NODE" ]; then
    for node in /sys/class/drm/renderD*/device; do
        dev_id="$(cat "$node/device" 2>/dev/null || true)"
        if [ "$dev_id" = "$VEGA8_PCI_ID" ]; then
            render_name="$(basename "$(dirname "$node")")"
            VEGA8_RENDER_NODE="/dev/dri/$render_name"
            break
        fi
    done
fi

if [ -z "$VEGA8_RENDER_NODE" ]; then
    echo "⚠  Could not auto-detect Vega 8 render node (PCI ID $VEGA8_PCI_ID)."
    echo "   Falling back to passing all /dev/dri devices — ROCm may pick the wrong GPU."
    echo "   Override: VEGA8_RENDER_NODE=/dev/dri/renderDXXX ./run-docker-rocm7.sh ..."
    EXTRA_DEVICES="--device=/dev/dri"
else
    echo "  Vega 8 render node: $VEGA8_RENDER_NODE"
    EXTRA_DEVICES="--device=$VEGA8_RENDER_NODE"
fi

# ── Auto-build image if missing ───────────────────────────────────────────────
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Docker image '$IMAGE_NAME' not found. Building (this will take ~20-40 min)..."
    docker build -t "$IMAGE_NAME" -f "$(dirname "$0")/Dockerfile.rocm7-vega" "$(dirname "$0")"
fi

echo "════════════════════════════════════════════════════════"
echo "  llama-server  ·  ROCm 7.2  ·  Vega 8 iGPU  (EXPERIMENTAL)"
echo "════════════════════════════════════════════════════════"
echo "  Model:  $MODEL_NAME"
echo "  Dir:    $MODEL_DIR → /models"
echo ""

# shellcheck disable=SC2086
docker run --rm -it \
  --device=/dev/kfd \
  $EXTRA_DEVICES \
  --group-add=video \
  --group-add=render \
  --ipc=host \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  -e ROCR_VISIBLE_DEVICES=0 \
  -e HIP_VISIBLE_DEVICES=0 \
  -v "$MODEL_DIR:/models:ro" \
  -p 8080:8080 \
  "$IMAGE_NAME" \
  --host 0.0.0.0 \
  -m "/models/$MODEL_NAME" \
  -fa 1 \
  -ngl 99 \
  "$@"
