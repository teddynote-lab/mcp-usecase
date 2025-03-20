from typing import List
import os

from dotenv import load_dotenv
from exa_py import Exa
from mcp.server.fastmcp import FastMCP

load_dotenv(override=True)

exa_api_key = os.getenv("EXA_API_KEY")
exa = Exa(api_key=exa_api_key)

websearch_config = {
    "parameters": {
        "default_num_results": 5,
        "include_domains": []
    }
}

mcp = FastMCP(
    name="websearch", 
    version="1.0.0",
    description="Web search capability using Exa API that provides real-time internet search results. Supports both basic and advanced search with filtering options including domain restrictions, text inclusion requirements, and date filtering. Returns formatted results with titles, URLs, publication dates, and content summaries."
)

def format_search_results(search_results):
    """검색 결과를 마크다운 형식으로 변환합니다.
    
    Args:
        search_results: Exa 검색 결과
        
    Returns:
        마크다운 형식의 문자열

    """

    if not search_results.results:
        return "No results found."

    markdown_results = "### Search Results:\n\n"

    for idx, result in enumerate(search_results.results, 1):
        title = result.title if hasattr(result, 'title') and result.title else "No title"
        url = result.url
        published_date = f" (Published: {result.published_date})" if hasattr(result, 'published_date') and result.published_date else ""
        
        markdown_results += f"**{idx}.** [{title}]({url}){published_date}\n"
        
        if hasattr(result, 'summary') and result.summary:
            markdown_results += f"> **Summary:** {result.summary}\n\n"
        else:
            markdown_results += "\n"
    
    return markdown_results
    
@mcp.tool()
async def search_web(query: str, num_results: int = None) -> str:
    """Exa API를 사용하여 웹을 검색하고 결과를 마크다운 형식으로 반환합니다.
    
    Args:
        query: 검색 쿼리
        num_results: 반환할 결과 수(기본 설정 무시)
    
    Returns:
        마크다운 형식의 검색 결과

    """

    try:
        search_args = {
            "num_results": num_results or websearch_config["parameters"]["default_num_results"]
        }
        
        search_results = exa.search_and_contents(
            query, 
            summary={"query": "Main points and key takeaways"},
            **search_args
        )
        
        return format_search_results(search_results)
    except Exception as e:
        return f"Exa 검색 중 오류가 발생했습니다: {e}"

@mcp.tool()
async def advanced_search_web(
    query: str, 
    num_results: int = None, 
    include_domains: List[str] = None, 
    include_text: str = None,
    max_age_days: int = None
) -> str:
    """Exa API를 사용한 고급 웹 검색으로 추가 필터링 옵션을 제공합니다.
    
    Args:
        query: 검색 쿼리
        num_results: 반환할 결과 수(기본 설정 무시)
        include_domains: 검색 결과에 포함할 도메인 목록
        include_text: 검색 결과에 반드시 포함되어야 하는 텍스트
        max_age_days: 결과의 최대 기간(일)
        
    Returns:
        마크다운 형식의 검색 결과

    """

    try:
        search_args = {
            "num_results": num_results or websearch_config["parameters"]["default_num_results"]
        }
        
        if include_domains:
            search_args["include_domains"] = include_domains

        elif websearch_config["parameters"]["include_domains"]:
            search_args["include_domains"] = websearch_config["parameters"]["include_domains"]
            
        if include_text:
            search_args["include_text"] = [include_text]
            
        if max_age_days:
            from datetime import datetime, timedelta
            start_date = (datetime.now() - timedelta(days=max_age_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
            search_args["start_published_date"] = start_date
        
        search_results = exa.search_and_contents(
            query, 
            summary={"query": "Main points and key takeaways"},
            **search_args
        )
        
        return format_search_results(search_results)
    except Exception as e:
        return f"Exa 고급 검색 중 오류가 발생했습니다: {e}"

@mcp.resource("help: hantaek@brain-crew.com")
def get_search_help() -> str:
    """웹 검색 도구에 대한 도움말을 제공합니다."""

    return """
            # 웹 검색 도구 사용 가이드
            
            Claude에게 Exa API를 통한 실시간 웹 검색 기능을 제공합니다.
            
            ## 기본 검색
            `search_web` 도구는 간단한 웹 검색을 수행합니다.
            - 매개변수: 
            - query: 검색 쿼리
            - num_results: 반환할 결과 수(선택 사항, 기본값: 5)
            
            ## 고급 검색
            `advanced_search_web` 도구는 추가 필터링 옵션이 있는 고급 검색을 제공합니다.
            - 매개변수:
            - query: 검색 쿼리
            - num_results: 반환할 결과 수(선택 사항)
            - include_domains: 검색 결과에 포함할 도메인 목록
            - include_text: 검색 결과에 반드시 포함되어야 하는 텍스트
            - max_age_days: 결과의 최대 기간(일)
            
            ## 사용 예시
            - 기본 검색: "최신 AI 발전 동향이 궁금해요"
            - 고급 검색: "특정 웹사이트에서만 검색하거나, 특정 텍스트가 포함된 결과만 찾고 싶을 때 사용하세요"

            """

if __name__ == "__main__":
    mcp.run()