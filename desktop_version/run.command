#!/bin/bash
# Created: 2026-06-02
# Launcher for Handwritten Digit Recognizer — Desktop Version

DIR="$(cd "$(dirname "$0")" && pwd)"

clear
echo "╔══════════════════════════════════════════╗"
echo "║   Digit Recognizer · Desktop Version     ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Python selection: prefer Homebrew (Tk 8.6+) over system (Tk 8.5) ──────────
PYTHON=""
for candidate in /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
    if command -v "$candidate" &>/dev/null; then
        TK_VER=$("$candidate" -c "import tkinter; print(tkinter.TkVersion)" 2>/dev/null)
        if [ -n "$TK_VER" ]; then
            PYTHON="$candidate"
            echo "Python: $($candidate --version 2>&1)  │  Tk $TK_VER"
            if python3 -c "exit(0 if float('$TK_VER') >= 8.6 else 1)" 2>/dev/null; then
                break
            fi
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo ""
    echo "[ERROR] Python with tkinter not found."
    echo "        Fix: brew install python-tk"
    echo ""
    read -rp "Press Enter to close…"
    exit 1
fi

# Warn if stuck on old Tk 8.5 (캔버스 렌더링 버그 가능)
TK_VER=$("$PYTHON" -c "import tkinter; print(tkinter.TkVersion)" 2>/dev/null)
if python3 -c "exit(0 if float('$TK_VER') < 8.6 else 1)" 2>/dev/null; then
    echo ""
    echo "⚠  Tk $TK_VER (구버전) — 그림이 안 그려지면 아래 명령으로 업그레이드 하세요:"
    echo "   brew install python-tk"
    echo ""
fi

# ── Dependency check ───────────────────────────────────────────────────────────
_check() { "$PYTHON" -c "import $1" 2>/dev/null || { echo "Installing $2…"; "$PYTHON" -m pip install "$2" --quiet --break-system-packages; }; }
_check numpy   numpy
_check PIL     pillow
_check sklearn scikit-learn

echo ""
echo "Launching…  (close the window to quit)"
echo ""

cd "$DIR" && "$PYTHON" -u app.py
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[ERROR] App exited with code $STATUS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    read -rp "Press Enter to close…"
fi
