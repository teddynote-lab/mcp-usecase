# 설치 가이드 (Installation Guide)

[English](#installation-guide) | [한국어](#설치-가이드-1)

## 설치 가이드

이 문서에서는 Quick-start Auto MCP 프로젝트의 설치 방법에 대해 상세히 설명합니다.

### 시스템 요구사항

- Python 3.11 이상
- Claude Desktop 또는 Cursor (MCP 지원 버전)
- Git (선택 사항)

### 설치 단계

#### 1. 저장소 복제

Git을 사용하여 저장소를 복제합니다.

```bash
git clone https://github.com/teddynote-lab/mcp.git
cd mcp
```

또는 GitHub에서 ZIP 파일로 다운로드하여 압축을 풀 수도 있습니다.

#### 2. 가상 환경 설정

##### uv 사용 (권장)

[uv](https://github.com/astral-sh/uv)는 더 빠른 파이썬 패키지 설치 및 환경 관리 도구입니다. 아직 설치하지 않았다면 먼저 설치해 주세요.

```bash
# uv 설치 (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# uv 설치 (Windows PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

uv를 사용하여 가상 환경을 설정하고 패키지를 설치합니다.

```bash
uv venv
uv pip install -r requirements.txt
```

##### pip 사용

기존 Python 도구를 사용하여 가상 환경을 설정할 수도 있습니다.

```bash
# 가상 환경 생성
python -m venv .venv

# 가상 환경 활성화 (Windows)
.venv\Scripts\activate

# 가상 환경 활성화 (macOS/Linux)
source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

#### 3. 환경 변수 설정

루트 디렉토리의 `.env.example`에 필요한 환경 변수를 설정하고 파일명을 `.env`로 바꿔주세요.

##### example1 (RAG)
```
OPENAI_API_KEY = "your-openai-api-key"
```

##### example2 (Dify External Knowledge API)
```
DIFY_API_ENDPOINT = http://localhost:8000/retrieval
DIFY_API_KEY = your-dify-api-key
DIFY_KNOWLEDGE_ID = your-knowledge-base-id
```

##### example3 (Dify Workflow)
```
DIFY_BASE_URL = https://api.dify.ai/v1
DIFY_APP_SK = your-dify-app-sk
```

##### example4 (Web Search)
```
EXA_API_KEY = your-exa-api-key
```

### 설치 확인

설치가 올바르게 이루어졌는지 확인하기 위해 다음 명령을 실행할 수 있습니다.

```bash
# 가상 환경이 활성화된 상태에서
cd example1
python auto_mcp_json.py
```

성공적으로 JSON 파일이 생성되면 설치가 완료된 것입니다.

---

## Installation Guide

This document provides detailed instructions for installing the Quick-start Auto MCP project.

### System Requirements

- Python 3.11 or higher
- Claude Desktop or Cursor (MCP supporting version)
- Git (optional)

### Installation Steps

#### 1. Clone the Repository

Clone the repository using Git:

```bash
git clone https://github.com/teddynote-lab/mcp.git
cd mcp
```

Alternatively, you can download and extract the ZIP file from GitHub.

#### 2. Set Up Virtual Environment

##### Using uv (recommended)

[uv](https://github.com/astral-sh/uv) is a faster Python package installer and environment manager. If you haven't installed it yet, install it first.

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install uv (Windows PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Set up a virtual environment and install packages using uv.

```bash
uv venv
uv pip install -r requirements.txt
```

##### Using pip

You can also set up a virtual environment using traditional Python tools.

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Activate virtual environment (macOS/Linux)
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

#### 3. Set Environment Variables for Each Example

Set the necessary environment variables in the .env.example file found in the root directory, then rename the file to .env.

##### example1 (RAG)
```
OPENAI_API_KEY = "your-openai-api-key"
```

##### example2 (Dify External Knowledge API)
```
DIFY_API_ENDPOINT = http://localhost:8000/retrieval
DIFY_API_KEY = your-dify-api-key
DIFY_KNOWLEDGE_ID = your-knowledge-base-id
```

##### example3 (Dify Workflow)
```
DIFY_BASE_URL = https://api.dify.ai/v1
DIFY_APP_SK = your-dify-app-sk
```

##### example4 (Web Search)
```
EXA_API_KEY = your-exa-api-key
```

### Verify Installation

To verify that the installation has been completed successfully, you can run the following command:

```bash
# With the virtual environment activated
cd example1
python auto_mcp_json.py
```

If a JSON file is successfully generated, the installation is complete.