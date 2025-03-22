# RAG(Retrieval Augmented Generation) 예제

[English](#rag-retrieval-augmented-generation-example) | [한국어](#rag-retrieval-augmented-generation-예제-1)

## RAG (Retrieval Augmented Generation) 예제

이 예제에서는 PDF 문서를 대상으로 키워드 검색, 시맨틱 검색, 하이브리드 검색 기능을 구현한 MCP 서버를 제공합니다.

### 기능

- **키워드 검색**: 문서 내에서 특정 키워드와 정확히 일치하는 내용을 검색합니다.
- **시맨틱 검색**: 임베딩 모델을 사용하여 의미적으로 유사한 내용을 검색합니다.
- **하이브리드 검색**: 키워드 검색과 시맨틱 검색을 결합하여 보다 정확한 결과를 제공합니다.

### 설정

다음 환경 변수를 루트 디렉토리의 `.env` 파일에 설정해야 합니다.

```
OPENAI_API_KEY = "sk-"
```

### 사용 방법

1. 환경 설정 확인
   ```bash
   # example1 디렉토리로 이동
   cd example1
   
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

Claude Desktop 또는 Cursor에서 다음과 같이 사용할 수 있습니다.

- **키워드 검색**: "키워드 검색으로 AI에 대한 정의를 찾아줄래?"
- **시맨틱 검색**: "시멘틱 검색을 해서 인공지능의 최근 발전 동향을 알려줘."
- **하이브리드 검색**: "하이브리드 검색을 해서 최근 LLM의 활용 사례를 알려줘."

### 구현 세부사항

`example1/mcp_server.py` 파일은 다음과 같은 주요 구성 요소로 이루어져 있습니다.

1. PDF 파일 경로 설정 및 RAG 체인 초기화
2. 검색 결과 포맷팅 함수
3. 키워드, 시맨틱, 하이브리드 검색 도구 정의

### 🚨 도구 Docstring 최적화

이 예제에서 제공하는 도구들이 Claude와 같은 AI 에이전트에 의해 효과적으로 사용되기 위해서는 명확하고 맥락적인 docstring을 작성하는 것이 중요합니다.

#### 왜 Docstring이 중요한가요?

`@mcp.tool()`로 도구를 정의할 때, 제공하는 docstring은 Claude가 도구를 이해하고 사용하는 방식에 직접적인 영향을 미칩니다. Claude는 다음과 같은 목적으로 docstring을 읽습니다.

1. **도구의 목적 이해**: Claude는 docstring을 분석하여 도구가 무엇을 하는지 파악합니다.
2. **사용 시점 결정**: Claude는 해당 도구를 사용해야 할 상황을 판단합니다.
3. **매개변수 형식 파악**: Claude는 필수 및 선택적 매개변수를 학습합니다.

사용자가 명시적으로 도구 이름을 언급하지 않더라도, 잘 작성된 docstring을 통해 Claude가 상황에 맞게 적절한 도구를 선택할 수 있습니다. 이는 더 자연스러운 대화 흐름과 최상의 결과를 얻는 데 필수적입니다.

#### 효과적인 Docstring 구조

최적의 결과를 위해 docstring을 다음과 같이 구성하세요:

```python
@mcp.tool()
async def your_tool_name(param1: str, param2: int = 5) -> str:
    """
    도구가 하는 일에 대한 짧은 설명 (1줄).
    결과 또는 출력 형식에 대한 자세한 내용 (1줄).
    이 도구를 사용해야 하는 상황에 대한 맥락적 힌트 (1줄).
    
    Parameters:
        param1: 첫 번째 매개변수 설명
        param2: 기본값이 있는 두 번째 매개변수 설명
    """
    # 함수 구현...
```



이러한 docstring을 통해 Claude는 다음과 같은 상황에서 지능적으로 도구를 선택할 수 있습니다.
- 사용자가 "문서에서 X의 정의는 무엇인가요?"라고 물으면 **keyword_search** 선택
- 사용자가 "문서에서 X 개념에 대해 설명해주세요"라고 물으면 **semantic_search** 선택
- 사용자가 "문서에서 X에 대해 무엇이라고 하나요?"라고 물으면 **hybrid_search** 선택

이처럼 사용자가 명시적으로 도구 이름을 언급하지 않더라도, 맥락적 힌트를 통해 Claude가 올바른 도구를 선택할 수 있습니다.

---

## RAG (Retrieval Augmented Generation) Example

This example provides an MCP server that implements keyword search, semantic search, and hybrid search functionality for PDF documents.

### Features

- **Keyword Search**: Searches for content that exactly matches specific keywords in documents.
- **Semantic Search**: Uses embedding models to search for semantically similar content.
- **Hybrid Search**: Combines keyword and semantic search to provide more accurate results.

### Configuration

Please ensure that the following environment variables are configured in the `.env` file at the root directory.

```
OPENAI_API_KEY = "sk-"
```

### Usage Instructions

1. Check environment configuration
   ```bash
   # Navigate to example1 directory
   cd example1
   
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

- **Keyword Search**: "Can you find the definition of AI using keyword search?"
- **Semantic Search**: "Could you use semantic search to tell me about recent developments in AI?"
- **Hybrid Search**: "Tell me about use cases of LLM using hybrid search."

### Implementation Details

The `example1/mcp_server.py` file consists of the following main components:

1. PDF file path setup and RAG chain initialization
2. Search results formatting function
3. Definition of keyword, semantic, and hybrid search tools

The vector database uses Chroma DB to efficiently store and search the contents of PDF documents.

### 🚨 Optimizing Tool Docstrings

For the tools in this example to be effectively used by AI agents like Claude, it's important to write clear and contextual docstrings.

#### Why Docstrings Matter

When defining tools with `@mcp.tool()`, the docstring you provide directly influences how Claude understands and uses the tool. Claude reads these docstrings to:

1. **Understand the tool's purpose**: Claude analyzes the docstring to know what the tool does
2. **Decide when to use it**: Claude determines which situations call for this specific tool
3. **Know how to format parameters**: Claude learns the required and optional parameters

Even when users don't explicitly mention the tool name, well-written docstrings allow Claude to select the appropriate tool based on the context. This is essential for a more natural conversation flow and optimal results.

#### Effective Docstring Structure

For optimal results, structure your docstrings like this:

```python
@mcp.tool()
async def your_tool_name(param1: str, param2: int = 5) -> str:
    """
    Short description of what the tool does (1 line).
    More details about the results or output format (1 line).
    Contextual hints about when to use this tool (1 line).
    
    Parameters:
        param1: Description of first parameter
        param2: Description of second parameter with default value
    """
    # Function implementation...
```

With these docstrings, Claude can intelligently choose:
- **keyword_search** when a user asks "What is the definition of X in the document?"
- **semantic_search** when a user asks "Tell me about the concept of X from the document"
- **hybrid_search** when a user asks "What does the document say about X?"

This way, even if users don't explicitly name the tool, Claude can select the right one through contextual hints.