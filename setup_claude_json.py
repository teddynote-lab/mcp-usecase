import os
import json
import argparse
from pathlib import Path
import platform
import sys

def get_claude_config_path():
    """OS에 따른 Claude Desktop 설정 파일 경로 반환"""
    if platform.system() == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif platform.system() == "Windows":
        return Path(os.environ.get("APPDATA")) / "Claude" / "claude_desktop_config.json"
    else:
        return None  # Linux 등 다른 OS

def create_claude_json(examples=None, output_path=None):
    """Claude Desktop 설정 JSON 생성"""
    
    project_root = Path(__file__).parent.absolute()
    python_path = sys.executable
    server_script = project_root / "mcp_server.py"
    
    config = {
        "mcpServers": {
            "claude-mcp": {
                "command": python_path,
                "args": [str(server_script)]
            }
        }
    }
    
    if examples:
        config["mcpServers"]["claude-mcp"]["args"].extend(["--examples"] + examples)
    
    json_path = Path(output_path) if output_path else project_root / "claude_config.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    return str(json_path)

def try_auto_setup(json_path):
    """가능한 경우 자동으로 설정 파일 설치 시도"""

    claude_path = get_claude_config_path()
    
    if not claude_path:
        return False, "지원되지 않는 OS입니다."
    
    if not claude_path.parent.exists():
        try:
            claude_path.parent.mkdir(parents=True, exist_ok=True)
        except:
            return False, f"디렉토리 생성 실패: {claude_path.parent}"
    
    try:
        with open(json_path, "r", encoding="utf-8") as src:
            config = json.load(src)
        
        with open(claude_path, "w", encoding="utf-8") as dest:
            json.dump(config, dest, indent=2)
        
        return True, str(claude_path)
    except Exception as e:
        return False, str(e)
    
def parse_examples(input_data):
    valid_choices = {"1", "2", "3"}
    
    if isinstance(input_data, str):
        examples = input_data.split()
    elif isinstance(input_data, list):
        examples = input_data
    else:
        return None

    if not examples or "0" in examples:
        return None
    elif set(examples).issubset(valid_choices):
        return examples
    else:
        print("잘못된 입력입니다. 모든 예제를 실행합니다.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Claude Desktop 설정 생성 도구")
    parser.add_argument("--examples", nargs="+", help="실행할 예제 번호 (미지정시 모든 예제)")
    parser.add_argument("--output", help="출력 파일 경로 (기본: 프로젝트 루트)")
    parser.add_argument("--auto", action="store_true", help="가능한 경우 자동으로 Claude 설정 폴더에 설치")
    
    args = parser.parse_args()
    
    if not args.examples and not sys.argv[1:]:
        print("=== Claude RAG 예제 선택 ===")
        print("여러 예제(1-3중에서)를 선택하려면 공백으로 구분해서 입력하세요. (예: 1 2)")
        print("1: 기본 RAG 구현")
        print("2: Dify 외부지식 API")
        print("3: Dify 워크플로우 연동")
        print("0: 모든 예제")
        
        choice = input("실행할 예제 번호를 선택하세요 (기본: 0): ").strip()
        examples = parse_examples(choice)        
        auto_setup = input("자동으로 Claude Desktop 설정을 업데이트할까요? (y/n, 기본: y): ").strip().lower() != "n"

    else:
        examples = parse_examples(args.examples)
        auto_setup = args.auto
    
    json_path = create_claude_json(examples, args.output)
    print(f"Claude Desktop 설정 파일이 생성되었습니다: {json_path}")
    
    if auto_setup:
        success, message = try_auto_setup(json_path)
        if success:
            print(f"Claude Desktop 설정이 자동으로 업데이트되었습니다: {message}")
            print("Claude Desktop을 재시작하여 변경사항을 적용하세요.")
        else:
            print(f"자동 설정 실패: {message}")
            print("\n수동 설정 방법:")
            claude_path = get_claude_config_path()
            print(f"1. 생성된 설정 파일({json_path})을 복사하세요.")
            print(f"2. Claude Desktop 설정 파일 위치({claude_path})에 붙여넣으세요.")
            print("3. Claude Desktop을 재시작하세요.")
    else:
        print("\n수동 설정 방법:")
        claude_path = get_claude_config_path()
        print(f"1. 생성된 설정 파일({json_path})을 복사하세요.")
        print(f"2. Claude Desktop 설정 파일 위치({claude_path})에 붙여넣으세요.")
        print("3. Claude Desktop을 재시작하세요.")

if __name__ == "__main__":
    main()