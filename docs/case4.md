# Web Search 예제

[English](#web-search-example) | [한국어](#web-search-예제-1)

## Web Search 예제

이 예제에서는 Tavily API를 활용하여 실시간 웹 검색 기능을 제공하는 MCP 서버를 구현합니다. 간단하고 효율적인 웹 검색 기능을 Claude에 제공합니다.

### 기능

- **웹 검색**: 간단한 쿼리로 웹 검색을 수행합니다.
- **결과 포맷팅**: 제목, URL, 발행 날짜, 내용 요약이 포함된 형식으로 결과를 반환합니다.

### 설정

다음 환경 변수를 루트 디렉토리의 `.env` 파일에 설정해야 합니다.

```
TAVILY_API_KEY = your-tavily-api-key
```

- `TAVILY_API_KEY`: Tavily API 키 (https://tavily.com/ 에서 발급 가능)

### 사용 방법

1. 환경 설정 확인
   ```bash
   # example4 디렉토리로 이동
   cd example4
   
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
최근 생성형 AI 기술 동향에 대해 검색해줘
```

```
인공지능 윤리에 대한 최신 논의를 검색해서 알려줘
```

### 구현 세부사항

`example4/mcp_server.py` 파일에는 다음과 같은 주요 구성 요소가 포함되어 있습니다:

1. Tavily API 연결 설정
2. 웹 검색 설정 구성
3. 검색 결과 포맷팅 함수
4. 웹 검색 도구
5. 도움말 리소스

이 구현은 Tavily API를 사용하여 검색 결과를 가져오고, 내용 요약을 자동으로 생성합니다. 검색 결과는 마크다운 형식으로 포맷팅되어 Claude에 표시됩니다.

---

## Web Search Example

This example implements an MCP server that provides real-time web search functionality using the Tavily API. It offers simple and efficient web search capabilities to Claude.

### Features

- **Web Search**: Performs web searches with simple queries.
- **Result Formatting**: Returns results in a format including titles, URLs, publication dates, and content summaries.

### Configuration

Please ensure that the following environment variables are configured in the `.env` file at the root directory.

```
TAVILY_API_KEY=your-tavily-api-key
```

- `TAVILY_API_KEY`: Tavily API key (can be obtained from https://tavily.com/)

### Usage Instructions

1. Check environment configuration
   ```bash
   # Navigate to example4 directory
   cd example4
   
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
Search for recent trends in generative AI technology
```

```
Find the latest discussions on AI ethics and provide the results
```

### Implementation Details

The `example4/mcp_server.py` file includes the following main components:

1. Tavily API connection setup
2. Web search configuration
3. Search results formatting function
4. Web search tool
5. Help resource

This implementation uses the Tavily API to fetch search results and automatically generate content summaries. The search results are formatted in markdown format and displayed in Claude.