#!/bin/bash
# Serve the chat UI with per-reply speed statistics.
#
# By default this starts NO model. It serves the page and nothing else, and the
# page talks to whatever llama-server you already have running -- started by
# run/start-llama-server.sh, by LM Studio's server, by a Docker backend, by hand.
# The server URL is editable in the page and remembered by the browser.
#
# That works because llama-server answers with Access-Control-Allow-Origin by
# default (verified 2026-09-15 against a server started with no CORS flags at
# all: preflight, POST and /props all echo the Origin back), so a page served on
# one port may call an API on another.
#
# Usage:
#   ./run/chat-ui.sh                        # serve the page on :8081, attach to :8080
#   UI_PORT=9000 ./run/chat-ui.sh           # serve it somewhere else
#   ./run/chat-ui.sh --with-server --rocm   # also start llama-server, same origin
#   ./run/chat-ui.sh --with-server --rocm --spec-type draft-mtp --spec-draft-n-max 2
#
# --with-server passes everything after it to run/start-llama-server.sh and adds
# `--path ui/`, so llama-server serves the page itself. Convenient for one
# command; not needed if a server is already up.
#
# Then open http://127.0.0.1:${UI_PORT:-8081}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
UI_DIR="$REPO_DIR/ui"
UI_PORT="${UI_PORT:-8081}"

if [ ! -f "$UI_DIR/index.html" ]; then
    echo "✗  $UI_DIR/index.html is missing — nothing to serve" >&2
    exit 1
fi

case "${1:-}" in
    --with-server)
        shift
        echo ""
        echo "  Starting llama-server, which will serve the UI itself."
        echo "  Chat UI:  http://127.0.0.1:${PORT:-8080}"
        echo ""
        exec "$SCRIPT_DIR/start-llama-server.sh" "$@" --path "$UI_DIR"
        ;;
    --help|-h)
        sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    "") : ;;
    *)
        echo "✗  Unknown option: $1" >&2
        echo "   This script serves the page; model flags belong after --with-server." >&2
        echo "   Try: $0 --help" >&2
        exit 1
        ;;
esac

echo ""
echo "  Chat UI:     http://127.0.0.1:$UI_PORT"
echo "  Expects a llama-server at http://127.0.0.1:8080 — change it in the page"
echo "  if yours listens elsewhere. Start one with: ./run/start-llama-server.sh"
echo ""

exec python3 -m http.server "$UI_PORT" --bind 127.0.0.1 --directory "$UI_DIR"
