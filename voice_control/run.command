#!/bin/bash
# Created: 2026-06-04
cd "$(dirname "$0")"

echo "=== 맥 음성 제어 ==="
echo ""

# Homebrew 확인
if ! command -v brew &>/dev/null; then
    echo "[설치] Homebrew 설치 중..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# portaudio 확인 (PyAudio 의존성)
if ! brew list portaudio &>/dev/null; then
    echo "[설치] portaudio 설치 중..."
    brew install portaudio
fi

# Python 패키지 확인
pip3 install -q SpeechRecognition pyaudio 2>/dev/null || {
    echo "[오류] 패키지 설치 실패. 아래 명령어를 수동으로 실행하세요:"
    echo "  brew install portaudio"
    echo "  pip3 install SpeechRecognition pyaudio"
    read -p "엔터를 누르면 종료합니다..."
    exit 1
}

echo "[준비완료] 마이크 권한이 필요합니다 (처음 실행 시 허용해주세요)"
echo ""

python3 voice_control.py
read -p "엔터를 누르면 창을 닫습니다..."
