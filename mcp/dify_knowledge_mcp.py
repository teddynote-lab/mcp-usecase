"""
Dify 외부 지식 API - MCP 연동 서버

이 스크립트는 기존 Dify 외부 지식 API를 Model Context Protocol(MCP)을 통해
Claude와 연결하는 별도의 서버를 구현합니다.

사용법:
1. 이 스크립트를 실행하여 MCP 서버 시작
2. MCP CLI를 통해 Claude Desktop에 서버 등록
   $ mcp install dify_knowledge_mcp.py --name "Dify 지식 검색"
"""

from mcp.server.fastmcp import FastMCP, Context
import httpx
import os
import json
import time
from typing import Dict, Optional
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API 설정
API_ENDPOINT = os.getenv("DIFY_API_ENDPOINT", "http://localhost:8000/retrieval")
API_KEY = os.getenv("DIFY_API_KEY", "dify-external-knowledge-api-key")
KNOWLEDGE_ID = os.getenv("DIFY_KNOWLEDGE_ID", "test-knowledge-base")

# 캐시 유효 시간 (초)
CACHE_TTL = 3600  # 1시간

# MCP 서버 생성
mcp = FastMCP("Dify Knowledge Bridge")

# 캐시 저장소
_cache: Dict[str, Dict] = {}
_cache_timestamps: Dict[str, float] = {}

def get_from_cache(key: str) -> Optional[Dict]:
    """캐시에서 결과를 가져옵니다."""
    if key not in _cache or key not in _cache_timestamps:
        return None
    
    # 캐시 만료 확인
    if time.time() - _cache_timestamps[key] > CACHE_TTL:
        del _cache[key]
        del _cache_timestamps[key]
        return None
    
    return _cache[key]

def add_to_cache(key: str, value: Dict) -> None:
    """결과를 캐시에 저장합니다."""
    _cache[key] = value
    _cache_timestamps[key] = time.time()

def format_search_results(data: Dict) -> str:
    """검색 결과를 가독성 높은 형태로 포맷팅합니다."""
    records = data.get("records", [])
    
    if not records:
        return "🔍 검색 결과가 없습니다."
    
    formatted_results = "📚 **검색 결과**\n\n"
    
    for i, record in enumerate(records):
        content = record.get("content", "")
        score = record.get("score", 0)
        title = record.get("title", f"결과 {i+1}")
        metadata = record.get("metadata", {})
        
        # 메타데이터에서 중요 정보 추출
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
    
    formatted_results += "ℹ️ 이 정보는 Dify 외부 지식 API를 통해 검색되었습니다."
    return formatted_results

@mcp.tool()
async def search_knowledge(
    query: str, 
    top_k: int = 5, 
    score_threshold: float = 0.5,
    search_method: str = "semantic_search",
    ctx: Context = None
) -> str:
    """
    Dify 외부 지식 API를 사용하여 문서에서 정보를 검색합니다.
    
    Parameters:
        query: 검색하려는 질문이나 키워드
        top_k: 반환할 최대 결과 수
        score_threshold: 결과로 포함할 최소 관련성 점수 (0.0-1.0)
    
    Returns:
        형식화된 검색 결과
    """
    if ctx:
        ctx.info(f"검색 쿼리: {query}")
        ctx.info(f"최대 결과 수: {top_k}")
        ctx.info(f"최소 점수: {score_threshold}")
    
    # 입력 유효성 검사
    if not query or not query.strip():
        return "오류: 검색 쿼리가 비어 있습니다."
    
    if top_k < 1:
        top_k = 1
    elif top_k > 20:
        top_k = 20
    
    if score_threshold < 0:
        score_threshold = 0
    elif score_threshold > 1:
        score_threshold = 1
    
    # 캐시 키 생성
    cache_key = f"{query}|{top_k}|{score_threshold}"
    
    # 캐시에서 확인
    cached_result = get_from_cache(cache_key)
    if cached_result:
        if ctx:
            ctx.info("캐시된 결과를 반환합니다.")
        return format_search_results(cached_result)
    
    # Dify API 호출
    try:
        if ctx:
            ctx.info(f"Dify API 호출 중: {API_ENDPOINT}")
        
        request_data = {
            "knowledge_id": KNOWLEDGE_ID,
            "query": query,
            "search_method": search_method,
            "retrieval_setting": {
                "top_k": top_k,
                "score_threshold": score_threshold
            }
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                API_ENDPOINT,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}"
                },
                json=request_data
            )
            
            # 응답 상태 코드 확인
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
                return f"🚫 **검색 실패**\n\n{error_message}"
            
            # 결과 처리
            try:
                data = response.json()
                
                # 결과 캐싱
                add_to_cache(cache_key, data)
                
                # 결과 포맷팅 및 반환
                return format_search_results(data)
                
            except json.JSONDecodeError:
                if ctx:
                    ctx.error("JSON 파싱 오류")
                return "🚫 **검색 실패**\n\nAPI 응답을 파싱할 수 없습니다."
            
    except httpx.RequestError as e:
        error_message = f"API 요청 오류: {str(e)}"
        if ctx:
            ctx.error(error_message)
        return f"🚫 **검색 실패**\n\n{error_message}"
    except Exception as e:
        error_message = f"예상치 못한 오류: {str(e)}"
        if ctx:
            ctx.error(error_message)
        return f"🚫 **검색 실패**\n\n{error_message}"

@mcp.tool()
async def check_knowledge_service(ctx: Context = None) -> str:
    """
    Dify 외부 지식 API 서비스의 연결 상태와 건강 상태를 확인합니다.
    이 도구는 서비스가 정상적으로 작동하는지 확인하는 데 사용합니다.
    """
    health_endpoint = API_ENDPOINT.replace("/retrieval", "/health")
    
    if ctx:
        ctx.info(f"서비스 건강 상태 확인 중: {health_endpoint}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # 먼저 헬스 엔드포인트 확인
            try:
                health_response = await client.get(health_endpoint)
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    
                    # 건강 상태 포맷팅
                    status = "✅ 정상" if health_data.get("status") == "healthy" else "⚠️ 비정상"
                    status_details = []
                    
                    for key, value in health_data.items():
                        if key == "status":
                            continue
                        status_details.append(f"- {key}: {'✅ 예' if value else '❌ 아니오'}")
                    
                    health_status = f"**건강 상태**: {status}\n\n" + "\n".join(status_details)
                else:
                    health_status = f"**건강 상태**: ⚠️ 확인 불가 (HTTP {health_response.status_code})"
            except:
                health_status = "**건강 상태**: ⚠️ 확인 불가 (엔드포인트 접근 실패)"
            
            # 기본 연결 테스트
            try:
                ping_response = await client.post(
                    API_ENDPOINT,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {API_KEY}"
                    },
                    json={
                        "knowledge_id": KNOWLEDGE_ID,
                        "query": "connection test",
                        "retrieval_setting": {
                            "top_k": 1,
                            "score_threshold": 0.1
                        }
                    },
                    timeout=5.0
                )
                connection_status = f"**연결 상태**: ✅ 연결됨 (HTTP {ping_response.status_code})"
            except Exception as e:
                connection_status = f"**연결 상태**: ❌ 연결 실패 ({str(e)})"
            
            # 설정 정보
            config_info = f"""
**서비스 설정**:
- API 엔드포인트: {API_ENDPOINT}
- 지식 베이스 ID: {KNOWLEDGE_ID}
- 캐시 TTL: {CACHE_TTL}초
"""
            
            return f"# Dify 외부 지식 서비스 상태\n\n{connection_status}\n\n{health_status}\n\n{config_info}"
            
    except Exception as e:
        return f"🚫 **상태 확인 실패**\n\n예상치 못한 오류가 발생했습니다: {str(e)}"

@mcp.resource("help://dify-knowledge")
def get_help() -> str:
    """Dify 지식 검색 도움말을 제공합니다."""
    return """
# Dify 외부 지식 검색 MCP 도구 사용법

이 MCP 도구는 Dify 외부 지식 API를 활용하여 PDF 문서에서 정보를 검색할 수 있게 해줍니다.

## 사용 가능한 도구

1. **search_knowledge** - 지식 베이스에서 정보 검색
   - `query`: 검색 쿼리
   - `top_k`: 반환할 최대 결과 수 (기본값: 5)
   - `score_threshold`: 최소 관련성 점수 (기본값: 0.5)

2. **check_knowledge_service** - 서비스 상태 확인

## 사용 예시

"대출 상환 조건에 대해 알려줘"와 같은 질문을 하면, Claude가 자동으로 외부 지식 베이스를 검색하여 관련 정보를 찾아줍니다.

구체적인 질문을 할수록 더 정확한 결과를 얻을 수 있습니다.
"""

# MCP 서버 실행
if __name__ == "__main__":
    print("Dify 지식 검색 MCP 서버를 시작합니다...")
    print(f"API 엔드포인트: {API_ENDPOINT}")
    print(f"지식 베이스 ID: {KNOWLEDGE_ID}")
    mcp.run()