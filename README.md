# 프로젝트 경로 자동 감지 및 설정 확인

`setup_claude_json.py` 스크립트는 다음과 같은 특징이 있습니다.

## 1. 경로 자동 감지

```python
# 프로젝트 루트 경로 (절대 경로) 자동 감지
project_root = Path(__file__).parent.absolute()

# 현재 Python 인터프리터 경로 자동 감지
python_path = sys.executable

# MCP 서버 스크립트 경로 자동 생성
server_script = project_root / "mcp_server.py"
```

이 코드는 사용자가 프로젝트를 어디에 클론했는지 상관없이
- 프로젝트 루트 디렉토리를 자동으로 찾음
- 현재 실행 중인 Python 인터프리터를 자동으로 찾음 (가상 환경 포함)
- 이를 기반으로 절대 경로를 계산하여 설정 파일 생성

## 2. OS 감지 및 설정 경로 찾기

```python
def get_claude_config_path():
    if platform.system() == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif platform.system() == "Windows":
        return Path(os.environ.get("APPDATA")) / "Claude" / "claude_desktop_config.json"
```

이 함수는:
- Windows인지 macOS인지 자동 감지
- 각 OS에 맞는 Claude Desktop 설정 파일 경로 반환

## 3. 클론 후 사용 시나리오

사용자가 레포지토리를 클론한 후의 흐름

1. **레포지토리 클론**:
   ```bash
   git clone https://github.com/yourusername/claude-rag.git
   cd claude-rag
   ```

2. **환경 설정** (가상 환경 및 패키지 설치)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Claude Desktop 설정**
   ```bash
   python setup_claude_json.py
   ```
   대화형 메뉴가 표시되고 예제 선택 가능

4. **자동 설치 성공 시**
   - "Claude Desktop 설정이 자동으로 업데이트되었습니다" 메시지 표시
   - 사용자는 Claude Desktop 재시작만 하면 됨

5. **자동 설치 실패 시**
   - 수동 설정 방법 안내
   - 생성된 `claude_config.json` 파일의 위치 표시
   - Claude Desktop 설정 파일 위치 표시
   - 복사-붙여넣기 지침 제공

이 스크립트는 사용자의 OS, 클론 위치, Python 환경을 자동으로 감지하여 모든 경로를 절대 경로로 변환합니다. 따라서 사용자는 어떤 환경에서도 쉽게 설정을 완료할 수 있습니다.