# Dify External Knowledge API 예제

[English](#dify-external-knowledge-api-example) | [한국어](#dify-external-knowledge-api-예제-1)

## Dify External Knowledge API 예제

이 예제에서는 Dify External Knowledge를 사용하여 지식 베이스에서 정보를 검색하는 MCP 서버를 제공합니다. 또한 SPRI 월간 AI 보고서를 기반으로 맞춤형 학습 가이드를 생성하는 `프롬프트 템플릿`도 포함되어 있습니다.

### 기능

- **다양한 검색 방법**: 시맨틱 검색, 키워드 검색, 하이브리드 검색을 지원합니다.
- **검색 결과 포맷팅**: 검색 결과를 가독성 있는 형태로 제공합니다.
- **AI 트렌드 학습 가이드**: SPRI 월간 AI 보고서를 기반으로 맞춤형 학습 가이드를 생성합니다.

### 설정

다음 환경 변수를 루트 디렉토리의 `.env` 파일에 설정해야 합니다.

```
DIFY_API_ENDPOINT = http://localhost:8000/retrieval
DIFY_API_KEY = your-dify-api-key
DIFY_KNOWLEDGE_ID = your-knowledge-base-id
```

- `DIFY_API_ENDPOINT`: Dify API 엔드포인트 URL
- `DIFY_API_KEY`: Dify API 키
- `DIFY_KNOWLEDGE_ID`: 검색할 지식 베이스 ID

### 사용 방법

1. 환경 설정 확인
   ```bash
   # example2 디렉토리로 이동
   cd example2
   
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

   # dify 외부지식 로컬서버 실행
   python dify_ek_server.py
   ```

3. Claude Desktop 또는 Cursor에 적용
   - 생성된 JSON 내용을 복사
   - Claude Desktop 또는 Cursor의 MCP 설정에 붙여넣기
   - 설정 저장 및 적용

### 사용 예시

Claude Desktop 또는 Cursor에서 다음과 같이 사용할 수 있습니다.

#### Dify 외부지식 검색
```
외부지식을 사용해서 최근 생성형 AI 기술 동향에 대해 검색해줘.
```

#### AI 트렌드 학습 가이드 생성
프롬프트 템플릿을 클릭해주세요.
```
Topic: LLM
Learning_level: 초급
Time_horizon: 중기
```

### 구현 세부사항

`example2/mcp_server.py` 파일에는 다음과 같은 주요 구성 요소가 포함되어 있습니다:

1. Dify API 연결 설정
2. 검색 결과 포맷팅 함수
3. 문서 검색 도구
4. AI 트렌드 학습 가이드 프롬프트 템플릿
5. 도움말 리소스

---

## Dify External Knowledge API Example

This example provides an MCP server that searches for information in a knowledge base using the Dify External Knowledge API. It also includes a prompt template for generating customized learning guides based on SPRI monthly AI reports.

### Features

- **Various Search Methods**: Supports semantic search, keyword search, and hybrid search.
- **Search Results Formatting**: Provides search results in a readable format.
- **AI Trends Learning Guide**: Generates customized learning guides based on SPRI monthly AI reports.

### Configuration

Please ensure that the following environment variables are configured in the `.env` file at the root directory.

```
DIFY_API_ENDPOINT = http://localhost:8000/retrieval
DIFY_API_KEY = your-dify-api-key
DIFY_KNOWLEDGE_ID = your-knowledge-base-id
```

- `DIFY_API_ENDPOINT`: Dify API endpoint URL
- `DIFY_API_KEY`: Dify API key
- `DIFY_KNOWLEDGE_ID`: Knowledge base ID to search

### Usage Instructions

1. Check environment configuration
   ```bash
   # Navigate to example2 directory
   cd example2
   
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

   # Run Dify external knowledge local server 
   python dify_ek_server.py
   ```

3. Apply to Claude Desktop or Cursor
   - Copy the generated JSON content
   - Paste it into the MCP settings of Claude Desktop or Cursor
   - Save and apply settings

### Usage Examples

You can use it in Claude Desktop or Cursor as follows:

#### External Knowledge Search with Dify
```
Use external knowledge to search for recent trends in LLM.
```

#### AI Trends Learning Guide Generation
Click the prompt template.
```
Topic: LLM  
Learning_level: Beginner  
Time_horizon: Mid-term  
```

### Implementation Details

The `example2/mcp_server.py` file includes the following main components:

1. Dify API connection setup
2. Search results formatting function
3. Document search tool
4. AI trends learning guide prompt template
5. Help resource