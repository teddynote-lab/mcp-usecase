[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/teddynote-lab-mcp-usecase-badge.png)](https://mseep.ai/app/teddynote-lab-mcp-usecase)

# Quick-start Auto MCP : All in one Claude Desktop and Cursor

[English](README.md) | [한국어](README_KR.md)

## Introduction

**Quick-start Auto MCP** is a tool that helps you easily and quickly register Anthropic's Model Context Protocol (MCP) in Claude Desktop and Cursor.

**Key advantages:**
1. **Quick Setup**: Add MCP functionality to Claude Desktop and Cursor simply by running a tool and copying/pasting the generated JSON file.
2. **Various Tools Provided**: We continuously update useful MCP tools. Stay up to date with your personalized toolkit by starring and following us. :)

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)
- [Author](#author)

## Features

- **RAG (Retrieval Augmented Generation)** - Keyword, semantic, and hybrid search functionality for PDF documents
- **Dify External Knowledge API** - Document search functionality via Dify's external knowledge API
- **Dify Workflow** - Execute and retrieve results from Dify Workflow
- **Web Search** - Real-time web search using Tavily API
- **Automatic JSON Generation** - Automatically generate MCP JSON files needed for Claude Desktop and Cursor

## Project Structure

```
.
├── case1                     # RAG example
├── case2                     # Dify External Knowledge API example
├── case3                     # Dify Workflow example
├── case4                     # Web Search example
├── data                      # Example data files
├── docs                      # Documentation folder
│   ├── case1.md           # case1 description 🚨 Includes tips for optimized tool invocation
│   ├── case2.md           # case2 description
│   ├── case3.md           # case3 description
│   ├── case4.md           # case4 description
│   └── installation.md    # Installation guide
├── .env.example              # .env example format
├── pyproject.toml            # Project settings
├── requirements.txt          # Required packages list
└── uv.lock                   # uv.lock
```

## Requirements

- Python >= 3.11
- Claude Desktop or Cursor (MCP supporting version)
- uv (recommended) or pip

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/teddynote-lab/mcp.git
cd mcp
```

### 2. Set up virtual environment

#### Using uv (recommended)
```bash
# macOS/Linux
uv venv
uv pip install -r requirements.txt
```

```bash
# Windows
uv venv
uv pip install -r requirements_windows.txt
```

#### Using pip
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
pip install -r requirements_windows.txt

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Preparing the PDF File

Plese prepare a PDF file required for RAG in the `./data` directory.

## Configuration

In order to execute each case, a `.env` file is required.
Please specify the necessary environment variables in the `.env.example` file located in the root directory, and rename it to `.env`.

### sites for configuring required environment variables for each case
- https://platform.openai.com/api-keys
- https://dify.ai/
- https://app.tavily.com/home

## Usage

### 1. Generate JSON File

Run the following command in each case directory to generate the necessary JSON file:

```bash
# Activate virtual environment

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Navigate to example directory
cd case1

# Generate JSON file
python auto_mcp_json.py
```

### 2. Register MCP in Claude Desktop/Cursor

1. Launch Claude Desktop or Cursor
2. Open MCP settings menu
3. Copy and paste the generated JSON content
4. Save and `restart` (If you're using Windows, we recommend fully closing the process via Task Manager and then restarting the application.)

> **Note**: When you run Claude Desktop or Cursor, the MCP server will automatically run with it. When you close the software, the MCP server will also terminate.

## Troubleshooting

Common issues and solutions:

- **MCP Server Connection Failure**: Check if the service is running properly and if there are no port conflicts. In particular, when applying case2, you must also run `dify_ek_server.py`.
- **API Key Errors**: Verify that environment variables are set correctly.
- **Virtual Environment Issues**: Ensure Python version is 3.11 or higher.

## License

[MIT LICENSE](LICENSE.md)

## Contributing

Contributions are always welcome! Please participate in the project through issue registration or pull requests. :)

## Contact

If you have questions or need help, please register an issue or contact:
dev@brain-crew.com

## Author
[Hantaek Lim](https://github.com/LHANTAEK)