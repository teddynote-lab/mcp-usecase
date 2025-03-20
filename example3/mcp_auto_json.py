import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

def get_env_variables():
    """환경 변수를 로드하고 필요한 변수를 딕셔너리로 반환합니다."""

    load_dotenv()
    
    required_vars = [
        "DIFY_BASE_URL",
        "DIFY_APP_SK"
    ]
    
    env_dict = {}
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            env_dict[var] = value
    
    return env_dict

def create_mcp_json():
    """MCP 서버 설정 JSON 파일을 생성합니다."""
    
    project_root = Path(__file__).parent.absolute()
    
    python_path = sys.executable
    
    server_script = project_root / "mcp_server.py"
    
    env_vars = get_env_variables()
    
    config = {
        "mcpServers": {
            "dify-workflow": {
                "command": python_path,
                "args": [str(server_script)],
                "env": env_vars
            }
        }
    }
    
    json_path = project_root / "mcp_config.json"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    print(f"MCP 설정 파일이 생성되었습니다: {json_path}")
    print(f"생성된 환경 변수: {', '.join(env_vars.keys())}")
    
    return str(json_path)

if __name__ == "__main__":
    create_mcp_json()