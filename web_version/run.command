#!/bin/bash
# Created: 2026-06-02 15:27
# Launcher for Handwritten Digit Recognizer — Web Version

DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=5001

clear
echo "╔══════════════════════════════════════════╗"
echo "║   Digit Recognizer · Web Version         ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Install missing packages (pip name → import name mapping, bash 3.2 compatible)
_check() { python3 -c "import $1" 2>/dev/null || { echo "Installing $2 …"; pip3 install "$2" --quiet; }; }
_check flask       flask
_check numpy       numpy
_check PIL         pillow
_check sklearn     scikit-learn

# Kill any previous instance on this port
OLD=$(lsof -ti tcp:$PORT 2>/dev/null)
[ -n "$OLD" ] && { echo "Stopping previous instance…"; kill "$OLD"; sleep 1; }

echo "Starting server on http://127.0.0.1:$PORT …"
cd "$DIR"
python3 -u app.py &
SERVER_PID=$!

echo "Waiting for server…"
for i in {1..30}; do
    curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/ 2>/dev/null | grep -q "200" && break
    sleep 1
done

echo "Opening browser…"
open "http://127.0.0.1:$PORT"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Running at http://127.0.0.1:$PORT"
echo "  Close this window to stop the server."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

trap "kill $SERVER_PID 2>/dev/null; exit 0" INT TERM EXIT
wait $SERVER_PID
