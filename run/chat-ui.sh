#!/bin/bash
# Chat UI with per-turn speed statistics, served by llama-server itself.
#
# llama-server's --path replaces its built-in web UI with a directory of static
# files, so the page is served from the same origin as the API. That means no
# CORS configuration, no second web server, and one URL to open.
#
# What the page shows, per reply: time to first token, prefill t/s, token
# generation t/s, prompt/completion token counts, wall time, and -- when the
# server runs with --spec-type draft-mtp -- the MTP draft acceptance rate.
# Numbers come from the server's own `timings` object, not from guesses on the
# client; the client only measures TTFT, which the server cannot see.
#
# Usage:
#   ./run/chat-ui.sh                          # Vulkan (the default backend)
#   ./run/chat-ui.sh --rocm                   # ROCm 7.2 baremetal
#   ./run/chat-ui.sh --cpu
#   MODEL=/path/to/model.gguf ./run/chat-ui.sh --rocm
#   ./run/chat-ui.sh --rocm --spec-type draft-mtp --spec-draft-n-max 2
#
# Environment: MODEL, CTX, PORT -- same as run/start-llama-server.sh.
#
# Then open http://127.0.0.1:${PORT:-8080}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
UI_DIR="$REPO_DIR/ui"

if [ ! -f "$UI_DIR/index.html" ]; then
    echo "✗  $UI_DIR/index.html is missing — the UI cannot be served" >&2
    exit 1
fi

case "${1:-}" in
    --rocm-docker)
        # The container has no bind mount for ui/, so --path would point at a
        # directory that does not exist inside it and llama-server would serve
        # nothing. The host backends have no such problem.
        echo "✗  --rocm-docker cannot serve the UI: the container cannot see $UI_DIR" >&2
        echo "   Use --rocm (baremetal), --vulkan or --cpu instead." >&2
        exit 1
        ;;
    --help|-h)
        sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
esac

echo ""
echo "  Chat UI:  http://127.0.0.1:${PORT:-8080}"
echo "  Stats come from the server's timings; MTP acceptance appears when the"
echo "  server is started with --spec-type draft-mtp."
echo ""

exec "$SCRIPT_DIR/start-llama-server.sh" "$@" --path "$UI_DIR"
