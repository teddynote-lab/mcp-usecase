# Dify Workflow 예제

[English](#dify-workflow-example) | [한국어](#dify-workflow-예제-1)

## Dify Workflow 예제

이 예제에서는 Dify Workflow API를 사용하여 워크플로우를 실행하고 결과를 가져오는 MCP 서버를 제공합니다. Dify의 워크플로우 기능을 통해 복잡한 AI 작업을 자동화하고 그 결과를 Claude에서 활용할 수 있습니다.

### 기능

- **Dify Workflow 실행**: 사용자 입력을 받아 Dify의 워크플로우를 실행합니다.
- **결과 반환**: 워크플로우 실행 결과를 Claude에 반환합니다.
- **에러 처리**: API 요청 및 응답 처리 과정에서 발생할 수 있는 에러를 적절하게 처리합니다.

### 설정

다음 환경 변수를 루트 디렉토리의 `.env` 파일에 설정해야 합니다.

```
DIFY_BASE_URL = https://api.dify.ai/v1
DIFY_APP_SK = your-dify-app-sk
```

- `DIFY_BASE_URL`: Dify API의 기본 URL
- `DIFY_APP_SK`: Dify 애플리케이션 시크릿 키

### 사용 방법

1. 환경 설정 확인
   ```bash
   # example3 디렉토리로 이동
   cd example3
   
   # 필요한 환경 변수 설정 확인
   # .env 파일이 올바르게 구성되었는지 확인하세요
   ```

2. JSON 파일 생성
   ```bash
   # 가상 환경 활성화 (아직 활성화하지 않은 경우)
   source ../.venv/bin/activate  # macOS/Linux
   ..\.venv\Scripts\activate      # Windows
   
   # JSON 파일 생성
   python auto_mcp_json.py
   ```

3. Claude Desktop 또는 Cursor에 적용
   - 생성된 JSON 내용을 복사
   - Claude Desktop 또는 Cursor의 MCP 설정에 붙여넣기
   - 설정 저장 및 적용

### 사용 예시

Claude Desktop 또는 Cursor에서 다음과 같이 사용할 수 있습니다:

```
Dify 워크플로우를 실행해서 "인공지능의 윤리적 문제"에 대한 분석을 해줘.
```

이 요청은 Dify 워크플로우에 "인공지능의 윤리적 문제"라는 입력을 전달하고, 워크플로우가 실행된 후 결과를 Claude에 표시합니다.

### 구현 세부사항

`example3/mcp_server.py` 파일에는 다음과 같은 주요 구성 요소가 포함되어 있습니다:

1. 환경 변수 로드
2. FastMCP 서버 초기화
3. Dify 워크플로우 실행 도구 정의
4. HTTP 요청 및 응답 처리
5. 에러 처리

이 구현은 Dify의 워크플로우 API를 사용하여 사용자 입력을 처리하고 결과를 반환합니다. 워크플로우는 Dify 플랫폼에서 미리 구성되어야 하며, 해당 워크플로우에 접근할 수 있는 권한이 있어야 합니다.

---

## Dify Workflow Example

This example provides an MCP server that executes workflows using the Dify Workflow API and retrieves the results. Through Dify's workflow functionality, you can automate complex AI tasks and utilize the results in Claude.

### Features

- **Execute Dify Workflow**: Executes Dify's workflow with user input.
- **Return Results**: Returns workflow execution results to Claude.
- **Error Handling**: Properly handles errors that may occur during API request and response processing.

### Configuration

Please ensure that the following environment variables are configured in the `.env` file at the root directory.

```
DIFY_BASE_URL = https://api.dify.ai/v1
DIFY_APP_SK = your-dify-app-sk
```

- `DIFY_BASE_URL`: Base URL for the Dify API
- `DIFY_APP_SK`: Secret key for your Dify application

### Usage Instructions

1. Check environment configuration
   ```bash
   # Navigate to example3 directory
   cd example3
   
   # Check the required environment variables
   # Make sure the .env file is properly configured
   ```

2. Generate JSON file
   ```bash
   # Activate virtual environment (if not already activated)
   source ../.venv/bin/activate  # macOS/Linux
   ..\.venv\Scripts\activate      # Windows
   
   # Generate JSON file
   python auto_mcp_json.py
   ```

3. Apply to Claude Desktop or Cursor
   - Copy the generated JSON content
   - Paste it into the MCP settings of Claude Desktop or Cursor
   - Save and apply settings

### Usage Examples

You can use it in Claude Desktop or Cursor as follows:

```
Execute a Dify workflow to analyze "ethical issues in artificial intelligence".
```

This request passes "ethical issues in artificial intelligence" as input to the Dify workflow, and after the workflow is executed, the results are displayed in Claude.

### Implementation Details

The `example3/mcp_server.py` file includes the following main components:

1. Load environment variables
2. Initialize FastMCP server
3. Define Dify workflow execution tool
4. Handle HTTP requests and responses
5. Error handling

This implementation uses Dify's workflow API to process user input and return results. The workflow must be pre-configured on the Dify platform, and you must have permission to access that workflow.