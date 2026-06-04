# Created: 2026-06-04
import speech_recognition as sr
import subprocess
import sys

# ── 맥 제어 함수 ──────────────────────────────────────────────
def run_apple(script):
    subprocess.run(['osascript', '-e', script])

def volume_up():
    run_apple('set volume output volume (output volume of (get volume settings) + 10)')
    print('볼륨 올림')

def volume_down():
    run_apple('set volume output volume (output volume of (get volume settings) - 10)')
    print('볼륨 내림')

def mute_toggle():
    run_apple('''
        set m to output muted of (get volume settings)
        if m then
            set volume output muted false
        else
            set volume output muted true
        end if
    ''')
    print('음소거 토글')

def open_app(name):
    subprocess.run(['open', '-a', name])
    print(f'{name} 실행')

def close_app(name):
    subprocess.run(['osascript', '-e', f'tell application "{name}" to quit'])
    print(f'{name} 종료')

def lock_screen():
    subprocess.run([
        'osascript', '-e',
        'tell application "System Events" to keystroke "q" using {control down, command down}'
    ])
    print('화면 잠금')

def take_screenshot():
    subprocess.run(['screencapture', '-i', '/tmp/screenshot.png'])
    subprocess.run(['open', '/tmp/screenshot.png'])
    print('스크린샷 저장됨')

def empty_trash():
    run_apple('tell application "Finder" to empty trash')
    print('휴지통 비움')

def show_desktop():
    run_apple('tell application "System Events" to key code 103 using {command down}')
    print('바탕화면 보기')

# ── 명령어 테이블 ──────────────────────────────────────────────
COMMANDS = {
    '볼륨 올려':      volume_up,
    '소리 올려':      volume_up,
    '볼륨 내려':      volume_down,
    '소리 내려':      volume_down,
    '음소거':         mute_toggle,
    '크롬 열어':      lambda: open_app('Google Chrome'),
    '사파리 열어':    lambda: open_app('Safari'),
    '파이어폭스 열어': lambda: open_app('Firefox'),
    '터미널 열어':    lambda: open_app('Terminal'),
    '파인더 열어':    lambda: open_app('Finder'),
    'vs코드 열어':    lambda: open_app('Visual Studio Code'),
    '크롬 닫아':      lambda: close_app('Google Chrome'),
    '사파리 닫아':    lambda: close_app('Safari'),
    '터미널 닫아':    lambda: close_app('Terminal'),
    '스크린샷':       take_screenshot,
    '화면 잠금':      lock_screen,
    '잠금':           lock_screen,
    '휴지통 비워':    empty_trash,
    '바탕화면':       show_desktop,
}

# ── 음성 인식 ──────────────────────────────────────────────────
recognizer = sr.Recognizer()

def listen_once():
    with sr.Microphone() as source:
        print('\n🎤 듣는 중... (말씀하세요)')
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
    return recognizer.recognize_google(audio, language='ko-KR')

def match_command(text):
    for keyword, action in COMMANDS.items():
        if keyword in text:
            return action
    return None

# ── 메인 루프 ──────────────────────────────────────────────────
def main():
    print('=' * 40)
    print('  맥 음성 제어 시작')
    print('  종료하려면 Ctrl+C')
    print('=' * 40)
    print('\n사용 가능한 명령어:')
    for k in COMMANDS:
        print(f'  · {k}')
    print()

    while True:
        try:
            text = listen_once()
            print(f'인식: "{text}"')

            action = match_command(text)
            if action:
                action()
            else:
                print('알 수 없는 명령어입니다')

        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            print('음성을 인식하지 못했습니다')
        except sr.RequestError as e:
            print(f'인터넷 연결을 확인하세요: {e}')
        except KeyboardInterrupt:
            print('\n종료합니다')
            sys.exit(0)

if __name__ == '__main__':
    main()
