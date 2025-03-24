import os
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


def get_env_variables():
    """Loads environment variables and returns them as a dictionary."""

    load_dotenv()

    required_vars = [
        "TAVILY_API_KEY",
    ]

    env_dict = {}

    for var in required_vars:
        value = os.getenv(var)
        if value:
            env_dict[var] = value

    return env_dict


def create_mcp_json():
    """Creates MCP server configuration JSON file."""

    project_root = Path(__file__).parent.absolute()

    # .venv python executable path
    if os.name == 'nt':  # Windows
        python_path = str(project_root.parent / ".venv" / "Scripts" / "python.exe")
    else:  # Mac, Ubuntu etc
        python_path = str(project_root.parent / ".venv" / "bin" / "python")

    server_script = project_root / "mcp_server.py"

    env_vars = get_env_variables()

    config = {
        "mcpServers": {
            "tavily-web-search": {
                "command": python_path,
                "args": [str(server_script)],
                "env": env_vars,
            }
        }
    }

    json_path = project_root / "mcp_config.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"MCP configuration file created: {json_path}")
    print(f"Generated environment variables: {', '.join(env_vars.keys())}")

    return str(json_path)


if __name__ == "__main__":
    create_mcp_json()
