# Claude와 Dify API 연동 흐름


## 1. 사용자 → Claude 질문 입력

사용자가 Claude에게 질문을 입력합니다:
```
"브레인크루의 사업 분야는 무엇인가요?"
```

## 2. Claude의 도구 호출 결정

Claude는 질문을 분석하고, 이 질문에 답하기 위해 외부 지식이 필요하다고 판단하면 MCP 도구를 사용하기로 결정합니다.

## 3. Claude → MCP 서버 통신

Claude는 `search_knowledge` 도구를 호출하기 위해 MCP 서버로 요청을 보냅니다:

```python
# 사용자 코드에서 정의된 도구 함수
@mcp.tool()
async def search_knowledge(
    query: str, 
    top_k: int = 5, 
    score_threshold: float = 0.5,
    search_method: str = "hybrid_search",
    ctx: Context = None
) -> str:
    """
    Dify 외부 지식 API를 사용하여 문서에서 정보를 검색합니다.
    """
    if ctx:
        ctx.info(f"검색 쿼리: {query}")
        ctx.info(f"최대 결과 수: {top_k}")
        ctx.info(f"최소 점수: {score_threshold}")
```

여기서 Claude는 내부적으로 함수의 매개변수를 지정하여 호출합니다. 이 코드의 특징은 기본 검색 방법으로 `hybrid_search`를 사용하고 있으며, `Context` 객체를 통해 로깅 기능을 제공한다는 점입니다.

## 4. 캐시 확인

도구 함수는 먼저 결과 캐싱을 통해 동일한 쿼리에 대한 반복 API 호출을 줄일 수 있습니다:

```python
# 캐시 확인 로직
cache_key = f"{query}|{top_k}|{score_threshold}"
cached_result = get_from_cache(cache_key)
if cached_result:
    if ctx:
        ctx.info("캐시된 결과를 반환합니다.")
    return format_search_results(cached_result)
```

이 부분은 반복 쿼리에 대한 성능 최적화를 위한 코드로, 캐시 키는 쿼리 내용과 검색 매개변수를 조합하여 생성됩니다.

## 5. Dify API 호출 준비

캐시에 결과가 없으면 Dify API를 직접 호출합니다:

```python
# API 요청 데이터 구성
request_data = {
    "knowledge_id": KNOWLEDGE_ID,
    "query": query,
    "search_method": search_method,
    "retrieval_setting": {
        "top_k": top_k,
        "score_threshold": score_threshold
    }
}
```

코드에서는 환경 변수나 상수에서 `KNOWLEDGE_ID`를 가져오고, Claude가 제공한 매개변수(쿼리, 검색 방법 등)를 사용하여 요청 본문을 구성합니다.

## 6. API 호출 실행

실제 API 호출은 `httpx` 라이브러리를 사용하여 비동기적으로 이루어집니다:

```python
# API 호출 수행
async with httpx.AsyncClient(timeout=30.0) as client:
    response = await client.post(
        API_ENDPOINT,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        },
        json=request_data
    )
```

여기서 API 엔드포인트와 인증 키는 파일 상단에 정의된 상수입니다. 코드에서 30초 타임아웃을 설정하여 응답 지연 시 너무 오래 기다리지 않도록 했습니다.

## 7. 오류 처리

API 호출 후 응답 상태 코드를 확인하여 오류 상황을 처리합니다:

```python
# 응답 상태 코드 확인 및 오류 처리
if response.status_code != 200:
    error_message = f"Dify API 오류: HTTP {response.status_code}"
    try:
        error_detail = response.json()
        if isinstance(error_detail, dict) and "error_msg" in error_detail:
            error_message += f" - {error_detail['error_msg']}"
    except:
        error_message += f" - {response.text[:100]}"
    
    if ctx:
        ctx.error(error_message)
    return f"검색 실패\n\n{error_message}"
```

이 부분은 강건한 오류 처리를 위한 코드로, API 응답이 실패하면 오류 메시지를 구성하고 Context를 통해 로깅합니다.

## 8. 결과 처리 및 캐싱

성공적인 응답을 받으면 결과를 처리하고 캐시에 저장합니다:

```python
# JSON 응답 처리
data = response.json()

# 결과 캐싱
add_to_cache(cache_key, data)

# 결과 포맷팅 및 반환
return format_search_results(data)
```

`add_to_cache` 함수를 호출하여 결과를 캐시에 저장하고, `format_search_results` 함수를 통해 결과를 사용자 친화적인 형식으로 변환합니다.

## 9. 결과 포맷팅

`format_search_results` 함수는 API 응답을 읽기 쉬운 텍스트로 변환합니다:

```python
def format_search_results(data: Dict) -> str:
    """검색 결과를 가독성 높은 형태로 포맷팅합니다."""
    records = data.get("records", [])
    
    if not records:
        return "검색 결과가 없습니다."
    
    formatted_results = "검색 결과\n\n"
    
    for i, record in enumerate(records):
        content = record.get("content", "")
        score = record.get("score", 0)
        title = record.get("title", f"결과 {i+1}")
        metadata = record.get("metadata", {})
        
        # 일부 메타데이터 추출
        source_info = []
        if "title" in metadata:
            source_info.append(f"파일: {os.path.basename(metadata['title'])}")
        if "page" in metadata:
            source_info.append(f"페이지: {metadata['page']}")
            
        source_text = " | ".join(source_info) if source_info else "출처 정보 없음"
        
        formatted_results += f"### {title} (관련도: {score:.2f})\n"
        formatted_results += f"*{source_text}*\n\n"
        formatted_results += f"{content}\n\n"
        formatted_results += "---\n\n"
    
    formatted_results += "이 정보는 Dify 외부 지식 API를 통해 검색되었습니다."

    return formatted_results
```

이 함수는 API 응답의 각 레코드에서 내용, 점수, 제목, 메타데이터를 추출하여 마크다운 형식으로 포맷팅합니다. 특히 메타데이터에서 파일명과 페이지 번호를 추출하여 출처 정보를 제공합니다.

## 10. Claude로 결과 반환

최종적으로 포맷팅된 결과가 Claude로 반환됩니다:

```
검색 결과

### 런어데이 기업교육 제안서 (관련도: 0.95)
*파일: test.pdf | 페이지: 2*

회사개요
회사명 브레인크루㈜
대표이사 이경록 설립일 2018.11.30
사업분야 AI교육, 컨설팅, 외주개발
...

---

### 런어데이 기업교육 제안서 (관련도: 0.85)
*파일: test.pdf | 페이지: 4*

사업분야
생성형 AI시대를 위한 맞춤형 교육과 솔루션 개발
AI교육 외주개발 및 컨설팅
...

---

이 정보는 Dify 외부 지식 API를 통해 검색되었습니다.
```

## 11. Claude의 최종 응답 생성

Claude는 이 검색 결과를 사용하여 사용자 질문에 대한 최종 답변을 생성합니다. 검색 결과의 내용을 바탕으로 브레인크루의 사업 분야에 대한 요약된 정보를 제공합니다.

## 추가 기능: 서비스 상태 확인

코드에는 `check_knowledge_service` 함수도 포함되어 있어 서비스 상태를 확인할 수 있습니다:

```python
@mcp.tool()
async def check_knowledge_service(ctx: Context = None) -> str:
    """
    Dify 외부 지식 API 서비스의 연결 상태를 확인합니다.
    """
    # ... 상태 확인 코드 ...
```

이 도구는 서비스 상태 확인에 사용되며, API 서버의 `/health` 엔드포인트와 기본 연결 테스트를 수행합니다.