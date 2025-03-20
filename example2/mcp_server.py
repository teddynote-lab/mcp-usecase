from dotenv import load_dotenv
import httpx
import os
import json
import time
from typing import Dict, Optional
from pathlib import Path

load_dotenv()

API_ENDPOINT = os.getenv("DIFY_API_ENDPOINT", "http://localhost:8000/retrieval")
API_KEY = os.getenv("DIFY_API_KEY", "dify-external-knowledge-api-key")
KNOWLEDGE_ID = os.getenv("DIFY_KNOWLEDGE_ID", "test-knowledge-base")

def format_search_results(data: Dict) -> str:
    """검색 결과를 가독성 높은 형태로 포맷팅합니다."""

    records = data.get("records", [])
    
    if not records:
        return "검색 결과가 없습니다."
    
    formatted_results = "# 검색 결과\n\n"
    
    for i, record in enumerate(records):
        content = record.get("content", "")
        score = record.get("score", 0)
        title = record.get("title", f"결과 {i+1}")
        metadata = record.get("metadata", {})
        
        # 메타데이터 있는 경우 추출
        source_info = []
        if "title" in metadata:
            source_info.append(f"파일: {os.path.basename(metadata['title'])}")
        elif "path" in metadata:
            source_info.append(f"파일: {os.path.basename(metadata['path'])}")
        if "page" in metadata:
            source_info.append(f"페이지: {metadata['page']}")
            
        source_text = " | ".join(source_info) if source_info else "출처 정보 없음"
        
        formatted_results += f"## {title} (관련도: {score:.2f})\n"
        formatted_results += f"{source_text}\n\n"
        formatted_results += f"{content}\n\n"
        formatted_results += "---\n\n"
    
    formatted_results += "이 정보는 Dify 외부 지식 API를 통해 검색되었습니다."
    return formatted_results

def register_tools(mcp):
    """MCP 서버에 도구 등록"""
    
    @mcp.tool()
    async def dify_ek_search(
        query: str, 
        top_k: int = 5, 
        score_threshold: float = 0.5,
        search_method: str = "hybrid_search",
        ctx = None
    ) -> str:
        """
        Dify 외부 지식 API를 사용하여 문서에서 정보를 검색합니다.
        
        Parameters:
            query: 검색하려는 질문이나 키워드
            top_k: 반환할 최대 결과 수
            score_threshold: 결과로 포함할 최소 관련성 점수 (0.0-1.0)
            search_method: 검색 방법(semantic_search, keyword_search, hybrid_search)
        
        Returns:
            검색 결과 문서 내용

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
                
                try:
                    data = response.json()
                    return format_search_results(data)
                    
                except json.JSONDecodeError:
                    if ctx:
                        ctx.error("JSON 파싱 오류")
                    return "검색 실패\n\nAPI 응답을 파싱할 수 없습니다."
                
        except httpx.RequestError as e:
            error_message = f"API 요청 오류: {str(e)}"
            if ctx:
                ctx.error(error_message)
            return f"검색 실패\n\n{error_message}"
        
        except Exception as e:
            error_message = f"예상치 못한 오류: {str(e)}"
            if ctx:
                ctx.error(error_message)
            return f"검색 실패\n\n{error_message}"

    @mcp.prompt()
    def ai_trend_learning_guide(
        topic: str = "",
        learning_level: str = "입문",
        time_horizon: str = "단기"
    ) -> str:
        """
        SPRI 월간 AI 레포트를 기반으로 AI 트렌드를 분석하고 맞춤형 학습 가이드를 제공합니다.
        
        Parameters:
            topic: 관심 있는 AI 주제 (선택사항 -> 예: "생성형 AI", "컴퓨터 비전", "자연어 처리" 등)
            learning_level: 학습자 수준 ("입문", "중급", "고급")
            time_horizon: 학습 계획 기간 ("단기", "중기", "장기")
        
        Returns:
            AI 트렌드 분석 및 학습 가이드 프롬프트

        """
        
        level_approaches = {
            "입문": "기본 개념과 원리 이해에 중점을 두고, 실습 위주의 학습 경로를 제안합니다.",
            "중급": "심화 개념과 실제 프로젝트 구현 방법에 중점을 두고, 응용 능력 향상을 위한 학습 경로를 제안합니다.",
            "고급": "최신 연구 동향과 고급 기술 구현에 중점을 두고, 혁신적 접근법과 전문성 강화를 위한 학습 경로를 제안합니다."
        }
        
        time_plans = {
            "단기": "1-3개월 내 습득 가능한 핵심 기술과 지식을 중심으로 집중적인 학습 계획을 제안합니다.",
            "중기": "3-6개월에 걸쳐 체계적으로 역량을 쌓을 수 있는 단계별 학습 계획을 제안합니다.",
            "장기": "6개월-1년 이상의 장기적 관점에서 전문성을 키울 수 있는 포괄적인 학습 계획을 제안합니다."
        }
        
        level_approach = level_approaches.get(learning_level, level_approaches["입문"])
        time_plan = time_plans.get(time_horizon, time_plans["단기"])
        
        output_template = f"""
        # {topic if topic else 'AI 트렌드'} 학습 가이드
        
        ## 1. 트렌드 분석
        - 주요 동향
        - 기술적 변화
        - 산업 영향
        
        ## 2. 핵심 지식 영역
        - 기본 개념
        - 핵심 기술
        - 주요 알고리즘/방법론
        
        ## 3. 학습 로드맵
        - 단계별 학습 계획
        - 추천 자료
        - 실습 프로젝트
        
        ## 4. 진로 및 활용 방안
        - 관련 직무/역할
        - 산업별 활용 사례
        - 미래 전망
        """
        
        # 최종 프롬프트 생성
        prompt = (
            f"당신은 AI 학습 가이드 전문가로, SPRI 월간 AI 레포트의 내용을 기반으로 최신 AI 트렌드를 분석하고 "
            f"맞춤형 학습 방향을 제시합니다.\n\n"
            
            f"## 학습자 프로필\n"
            f"- 수준: {learning_level} ({level_approach})\n"
            f"- 학습 계획: {time_horizon} ({time_plan})\n\n"
            
            f"## 분석 대상\n"
            f"SPRI에서 제공하는 월간 AI 레포트 3월호를 기반으로 분석해주세요. "
            f"{'특히 ' + topic + '에 관련된 내용에 중점을 두어 분석해주세요.' if topic else '전반적인 AI 트렌드를 분석해주세요.'}\n\n"
            
            f"## 제공해야 할 정보\n"
            f"1. 최신 AI 트렌드 요약 및 중요성\n"
            f"2. 해당 분야의 핵심 지식과 기술 요소\n"
            f"3. 단계별 학습 계획 및 추천 자료\n"
            f"4. 실질적인 응용 방안 및 진로 제안\n\n"
            
            f"다음 구조에 맞춰 분석 결과를 작성해주세요:\n\n{output_template}\n\n"
            
            f"레포트 내용을 검색하여 실질적이고 구체적인 정보를 제공해주세요. 최신 트렌드에 맞는 학습 방향을 "
            f"제시하고, 학습자가 쉽게 따라할 수 있는 실용적인 가이드를 작성해주세요."
        )
        
        return prompt
    
    @mcp.resource("help: hantaek@brain-crew.com")
    def get_help() -> str:
        """Claude Desktop에서 Dify 지식 검색 도움말을 제공합니다."""

        return """
                # Dify 외부 지식 검색 MCP 도구 사용법

                클로드가 Dify 외부 지식 API를 활용하여 문서에서 정보를 검색할 수 있게 해줍니다.

                ## 사용 가능한 도구

                1. search_knowledge - 지식 베이스에서 정보 검색
                - `query`: 검색 쿼리
                - `top_k`: 반환할 최대 결과 수 (기본값: 5)
                - `score_threshold`: 최소 관련성 점수 (기본값: 0.5)
                - `search_method`: 검색 방법(semantic, keyword, hybrid)(기본값: hybrid)

                2. prompt(ai_trend_learning_guide)
                - `topic`: 관심 있는 AI 주제 (선택사항 -> 예: "생성형 AI", "컴퓨터 비전", "자연어 처리" 등)
                - `learning_level`: 학습자 수준 ("입문", "중급", "고급")
                - `time_horizon`: 학습 계획 기간 ("단기", "중기", "장기")

                """