#!/bin/bash
# Launcher for Handwritten Digit Recognizer
# Double-click this file in Finder to start the app.

DIR="$(cd "$(dirname "$0")" && pwd)"
APP="$DIR/digit_recognizer.py"
PORT=5001

# ── terminal appearance ──────────────────────────────────────────────────────
clear
echo "╔══════════════════════════════════════════╗"
echo "║   Handwritten Digit Recognizer           ║"
echo "║   손글씨 숫자 인식기                        ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── check for python3 ────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Please install Python 3 first."
    echo "        https://www.python.org/downloads/"
    read -p "Press Enter to exit..."
    exit 1
fi

# ── install missing packages ─────────────────────────────────────────────────
REQUIRED=(flask numpy pillow scikit-learn)
MISSING=()
for pkg in "${REQUIRED[@]}"; do
    python3 -c "import ${pkg//-/_}" 2>/dev/null || MISSING+=("$pkg")
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Installing missing packages: ${MISSING[*]}"
    echo ""
    pip3 install "${MISSING[@]}" --quiet || {
        echo "[ERROR] Package installation failed."
        read -p "Press Enter to exit..."
        exit 1
    }
    echo ""
fi

# ── kill any previous instance on this port ──────────────────────────────────
OLD_PID=$(lsof -ti tcp:$PORT 2>/dev/null)
if [ -n "$OLD_PID" ]; then
    echo "Stopping previous instance (PID $OLD_PID)..."
    kill "$OLD_PID" 2>/dev/null
    sleep 1
fi

# ── start the server ─────────────────────────────────────────────────────────
echo "Starting server on http://127.0.0.1:$PORT ..."
echo ""
cd "$DIR"
python3 -u digit_recognizer.py &
SERVER_PID=$!

# ── wait for server to be ready, then open browser ───────────────────────────
echo "Waiting for server to be ready..."
for i in {1..30}; do
    if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/ 2>/dev/null | grep -q "200"; then
        echo "Server is ready!"
        break
    fi
    sleep 1
done

echo ""
echo "Opening browser → http://127.0.0.1:$PORT"
open "http://127.0.0.1:$PORT"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  App is running. Close this window to stop."
echo "  앱 실행 중. 이 창을 닫으면 종료됩니다."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Keep terminal open; shut down server when user closes the window
trap "echo ''; echo 'Shutting down...'; kill $SERVER_PID 2>/dev/null; exit 0" INT TERM EXIT
wait $SERVER_PID
