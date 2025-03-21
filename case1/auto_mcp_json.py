import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

def get_env_variables():
    """
    Load environment variables and return required variables as a dictionary.
    
    Returns:
        dict: Dictionary containing environment variables and default configuration values
    """

    load_dotenv()
    
    required_vars = [
        "OPENAI_API_KEY",
    ]

    config_vars = {
        "DEFAULT_TOP_K": "5"
    }
    
    env_dict = {}
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            env_dict[var] = value

    env_dict.update(config_vars)
    
    return env_dict

def create_mcp_json():
    """
    Create a Model Context Protocol (MCP) server configuration JSON file.
    
    This function generates a configuration file that defines how the MCP server
    should be launched, including the Python interpreter path, server script location,
    and necessary environment variables.
    
    Returns:
        str: Path to the created JSON configuration file
    """
    
    project_root = Path(__file__).parent.absolute()
    
    python_path = sys.executable
    
    server_script = project_root / "mcp_server.py"
    
    env_vars = get_env_variables()
    
    config = {
        "mcpServers": {
            "rag-mcp": {
                "command": python_path,
                "args": [str(server_script)],
                "env": env_vars
            }
        }
    }
    
    json_path = project_root / "mcp_config.json"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    print(f"MCP configuration file has been created: {json_path}")
    print(f"Generated environment variables: {', '.join(env_vars.keys())}")
    
    return str(json_path)

if __name__ == "__main__":
    create_mcp_json()