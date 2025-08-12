import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv


def select_mcp_file():
    """
    Find all files starting with 'mcp' and ending with '.py' and let user select one.
    
    Returns:
        Path: Selected file path or None if cancelled
    """
    project_root = Path(__file__).parent.absolute()
    
    # Find all mcp*.py files
    mcp_files = sorted(project_root.glob("mcp*.py"))
    
    if not mcp_files:
        print("No files starting with 'mcp' and ending with '.py' found.")
        return None
    
    print("\nAvailable MCP server files:")
    print("-" * 40)
    for i, file_path in enumerate(mcp_files, 1):
        print(f"{i}. {file_path.name}")
    print("-" * 40)
    
    while True:
        try:
            choice = input("\nSelect file number (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                print("Cancelled.")
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(mcp_files):
                selected_file = mcp_files[choice_num - 1]
                print(f"\nSelected: {selected_file.name}")
                return selected_file
            else:
                print(f"Please enter a number between 1 and {len(mcp_files)}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")


def create_mcp_json(server_script=None):
    """
    Create a Model Context Protocol (MCP) server configuration JSON file.

    This function generates a configuration file that defines how the MCP server
    should be launched, including the Python interpreter path, server script location,
    and necessary environment variables.
    
    Args:
        server_script: Path to the MCP server script. If None, user will be prompted to select.

    Returns:
        str: Path to the created JSON configuration file
    """

    project_root = Path(__file__).parent.absolute()
    
    # If no server script provided, let user select
    if server_script is None:
        server_script = select_mcp_file()
        if server_script is None:
            return None

    # .venv python executable path
    if os.name == "nt":  # Windows
        python_path = str(project_root.parent / ".venv" / "Scripts" / "python.exe")
    else:  # Mac, Ubuntu etc
        python_path = str(project_root.parent / ".venv" / "bin" / "python")

    # Generate config name based on script name (remove mcp_ prefix and .py suffix)
    config_name = server_script.stem
    if config_name.startswith("mcp_"):
        config_name = config_name[4:]  # Remove 'mcp_' prefix
    
    config = {
        "mcpServers": {
            config_name: {"command": python_path, "args": [str(server_script)]}
        }
    }

    json_path = project_root / f"mcp_config_{config_name}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nMCP configuration file has been created: {json_path}")
    print(f"Server name: {config_name}")

    return str(json_path)


if __name__ == "__main__":
    create_mcp_json()
