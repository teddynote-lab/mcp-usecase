from mcp.server.fastmcp import FastMCP, Context
import httpx
import os
import json
import time
from typing import Dict, Optional

"""
Dify 외부 지식 API - MCP 연동 서버
Claude와 연결하는 Model Context Protocol(MCP) 서버를 구현하는 스크립트입니다.
"""

# 환경변수 설정
API_ENDPOINT = "http://localhost:8000/retrieval"
API_KEY = "dify-external-knowledge-api-key"
KNOWLEDGE_ID = "test-knowledge-base"

# 캐시 유효 시간(초)
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
    
    Parameters:
        query: 검색하려는 질문이나 키워드
        top_k: 반환할 최대 결과 수
        score_threshold: 결과로 포함할 최소 관련성 점수 (0.0-1.0)
        search_method: 검색 방법(semantic, keyword, hybrid)
    
    Returns:
        검색 결과
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
                return f"검색 실패\n\n{error_message}"
            
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


@mcp.tool()
async def check_knowledge_service(ctx: Context = None) -> str:
    """
    Dify 외부 지식 API 서비스의 연결 상태를 확인합니다.
    외부지식 서비스가 정상적으로 작동하는지 확인하는 데 사용됩니다.
    """

    health_endpoint = API_ENDPOINT.replace("/retrieval", "/health")
    
    if ctx:
        ctx.info(f"서비스 건강 상태 확인 중: {health_endpoint}")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                health_response = await client.get(health_endpoint)

                if health_response.status_code == 200:
                    health_data = health_response.json()
                    status = "정상" if health_data.get("status") == "healthy" else "비정상"
                    status_details = []
                    
                    for key, value in health_data.items():
                        if key == "status":
                            continue
                        status_details.append(f"- {key}: {'예' if value else '아니오'}")
                    
                    health_status = f"건강 상태: {status}\n\n" + "\n".join(status_details)

                else:
                    health_status = f"건강 상태: 확인 불가 (HTTP {health_response.status_code})"

            except:
                health_status = "건강 상태: 확인 불가 (엔드포인트 접근 실패)"
            
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
                connection_status = f"연결 상태: 연결됨 (HTTP {ping_response.status_code})"

            except Exception as e:
                connection_status = f"연결 상태: 연결 실패 ({str(e)})"
            
            # 설정 정보
            config_info = f"""
                            서비스 설정:
                            - API 엔드포인트: {API_ENDPOINT}
                            - 지식 베이스 ID: {KNOWLEDGE_ID}
                            - 캐시 TTL: {CACHE_TTL}초
                            """
            
            return f"# Dify 외부 지식 서비스 상태\n\n{connection_status}\n\n{health_status}\n\n{config_info}"
            
    except Exception as e:

        return f"상태 확인 실패\n\n예상치 못한 오류가 발생했습니다: {str(e)}"


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

            2. check_knowledge_service - 서비스 상태 확인

            ## 사용 예시

            "브레인크루가 위치한 곳은 어디고 번호는 뭐야?"와 같은 질문을 하면, Claude가 자동으로 외부 지식 베이스를 검색하여 관련 정보를 찾아줍니다.

            구체적인 질문을 할수록 더 정확한 결과를 얻을 수 있습니다.
            """


@mcp.prompt()
def company_strategy_analysis(
    company_name: str = "",
    focus_areas: str = "전체",
    output_format: str = "보고서"
) -> str:
    """
    회사 데이터를 바탕으로 종합적인 전략 분석 및 레포트를 생성하는 프롬프트입니다.
    
    Parameters:
        company_name: 분석할 회사 이름 (비어있으면 지식 베이스의 회사 정보 사용)
        focus_areas: 분석할 특정 영역 (예: "마케팅,인사,재무" 또는 "전체")
        output_format: 출력 형식 ("보고서", "프레젠테이션", "요약", "전략제안")
    
    Returns:
        회사 분석을 위한 프롬프트 문자열
    """
    
    # 부서별 분석 지침
    department_guidelines = {
        "경영/전략": "회사의 비전, 미션, 핵심 전략 방향 및 장기적 목표를 분석합니다. 경쟁사 대비 포지셔닝과 시장 기회를 평가합니다.",
        "재무": "재무 상태, 수익성, 투자 효율성, 재무 리스크를 분석합니다. 비용 구조 최적화와 재무적 성장 방안을 제안합니다.",
        "마케팅/영업": "마케팅 전략, 브랜드 포지셔닝, 고객 세그먼트, 영업 채널의 효율성을 평가합니다. 시장 확대 및 고객 확보 전략을 제안합니다.",
        "인사/조직": "인재 채용 및 유지 전략, 조직 구조, 기업 문화, 성과 관리 시스템을 분석합니다. 조직 역량 강화 방안을 제안합니다.",
        "연구개발": "R&D 투자, 혁신 역량, 기술 로드맵, 지적 재산권 관리를 평가합니다. 기술 경쟁력 강화 전략을 제안합니다.",
        "운영/생산": "운영 효율성, 공급망 관리, 품질 관리, 생산성을 분석합니다. 비용 절감 및 운영 최적화 방안을 제안합니다.",
        "IT/디지털": "IT 인프라, 디지털 역량, 데이터 활용도, 디지털 트랜스포메이션 수준을 평가합니다. 디지털 혁신 전략을 제안합니다.",
        "법무/컴플라이언스": "법적 리스크, 규제 준수 상태, 계약 관리, 지적 재산권 보호를 검토합니다. 법적 리스크 최소화 방안을 제안합니다."
    }
    
    # 출력 형식별 구조
    format_structures = {
        "보고서": (
            "# {company_name} 종합 전략 분석 보고서\n\n"
            "## 1. 개요\n- 회사 현황 요약\n- 분석 범위 및 방법론\n- 주요 발견 사항\n\n"
            "## 2. 내부 역량 분석\n- 조직 구조 및 인력\n- 핵심 역량 및 자원\n- 내부 프로세스 효율성\n\n"
            "## 3. 외부 환경 분석\n- 시장 동향 및 기회\n- 경쟁사 분석\n- 규제 및 법적 환경\n\n"
            "## 4. 부서별 분석 및 제언\n{department_sections}\n\n"
            "## 5. 통합 전략 방향\n- 단기(1년) 추진 과제\n- 중기(2-3년) 전략 방향\n- 장기(5년) 비전 달성 방안\n\n"
            "## 6. 실행 계획 및 KPI\n- 주요 실행 과제\n- 성과 측정 지표\n- 모니터링 체계\n\n"
            "## 7. 결론 및 향후 과제"
        ),
        "프레젠테이션": (
            "# {company_name} 전략 분석 프레젠테이션\n\n"
            "## 슬라이드 1: 개요\n- 분석 목적 및 범위\n- 주요 인사이트\n\n"
            "## 슬라이드 2: 회사 현황\n- 핵심 지표 및 현재 상태\n\n"
            "## 슬라이드 3-4: 내부 역량 & 외부 환경\n- SWOT 분석\n\n"
            "## 슬라이드 5-12: 부서별 분석\n{department_sections}\n\n"
            "## 슬라이드 13-15: 전략 방향 및 실행계획\n- 단기/중기/장기 전략\n- 주요 실행과제\n\n"
            "## 슬라이드 16: Q&A"
        ),
        "요약": (
            "# {company_name} 전략 분석 요약\n\n"
            "## 현황 요약\n- 회사 현재 상태\n- 핵심 성과 지표\n\n"
            "## 주요 과제\n- 직면한 도전 과제\n- 시장 기회\n\n"
            "## 부서별 핵심 인사이트\n{department_sections}\n\n"
            "## 우선 추진 과제\n- Top 3 전략적 이니셔티브\n- 예상 효과"
        ),
        "전략제안": (
            "# {company_name}을 위한 전략 제안서\n\n"
            "## 현재 상황 진단\n- 회사의 현 위치\n- 핵심 도전 과제\n\n"
            "## 전략적 방향성\n- 비전 및 목표\n- 핵심 전략 축\n\n"
            "## 부서별 전략 제안\n{department_sections}\n\n"
            "## 실행 로드맵\n- 단계별 추진 계획\n- 필요 자원\n\n"
            "## 기대 효과\n- 정량적/정성적 효과\n- 투자 대비 효익 분석"
        )
    }
    
    # 분석할 부서 선택
    selected_departments = []
    if focus_areas.lower() == "전체":
        selected_departments = list(department_guidelines.keys())
    else:
        for area in focus_areas.split(','):
            area = area.strip()
            # 부분 일치하는 부서 찾기
            for dept in department_guidelines:
                if area in dept:
                    selected_departments.append(dept)
    
    # 없을 경우 기본값
    if not selected_departments:
        selected_departments = ["경영/전략", "마케팅/영업", "재무"]
    
    # 부서 섹션 생성
    department_sections = ""
    for dept in selected_departments:
        department_sections += f"### {dept}\n- 현황 및 문제점\n- 개선 방향\n- 실행 계획\n\n"
    
    # 출력 형식 선택 및 템플릿 생성
    format_template = format_structures.get(output_format, format_structures["보고서"])
    format_template = format_template.format(
        company_name="{회사명}" if not company_name else company_name,
        department_sections=department_sections
    )
    
    # 최종 프롬프트 문자열 생성
    prompt = (
        f"당신은 기업 전략 컨설턴트로, 회사 정보를 바탕으로 종합적인 분석과 전략적 제언을 제공합니다. "
        f"각 부서의 고유한 니즈와 도전 과제를 이해하고, 데이터 기반의 실용적인 통찰력을 제공하세요. "
        f"구체적인 실행 방안과 측정 가능한 성과 지표를 포함한 권장사항을 제시하세요.\n\n"
    )
    
    # 부서별 지침 추가
    prompt += "## 부서별 분석 지침\n\n"
    for dept in selected_departments:
        if dept in department_guidelines:
            prompt += f"### {dept}\n{department_guidelines[dept]}\n\n"
    
    # 출력 형식 지침 추가
    prompt += f"## 출력 형식\n\n다음 구조에 맞춰 분석 결과를 작성하세요:\n\n{format_template}\n\n"
    
    # 분석 지시 추가
    prompt += f"{company_name if company_name else '회사'} 데이터를 분석하여 "
    
    if focus_areas.lower() != "전체":
        prompt += f"{focus_areas} 부서를 중심으로 "
    
    prompt += f"{output_format} 형식의 종합적인 전략 분석을 제공해주세요."
    
    # 검색 지시 추가
    prompt += f"\n\n회사 정보는 지식 베이스에서 '{company_name if company_name else '회사'}'에 관한 데이터를 검색하여 활용해주세요."
    
    return prompt


@mcp.prompt()
def department_collaboration_analysis() -> str:
    """
    회사 내 부서 간 협업 및 커뮤니케이션 분석을 위한 프롬프트를 생성합니다.
    각 부서의 연결점과 협업 개선 방안을 제시합니다.
    """
    return """
            당신은 조직 효율성 및 부서 간 협업 전문가입니다. 회사소개서와 부서 정보를 바탕으로 부서 간 협업 패턴을 분석하고 개선 방안을 제시해주세요.

            다음 구조에 따라 분석을 진행해주세요:

            # 부서 간 협업 분석 및 개선 방안

            ## 1. 현재 조직 구조 분석
            - 주요 부서 및 기능 요약
            - 현재 보고 체계 및 의사결정 프로세스
            - 정보 흐름 맵핑

            ## 2. 협업 현황 진단
            - 부서 간 상호작용 빈도 및 품질
            - 주요 협업 통로 및 장벽
            - 의사결정 지연 또는 병목 지점
            - 중복 업무 및 책임 불명확 영역

            ## 3. 부서별 협업 니즈 및 기회
            [각 부서별로 다른 부서와의 핵심 협업 포인트 분석]

            ## 4. 시너지 극대화 전략
            - 조직 구조 최적화 제안
            - 협업 프로세스 개선안
            - 커뮤니케이션 채널 강화 방안
            - 통합 성과 관리 체계

            ## 5. 실행 계획
            - 단기 (1-3개월) 개선 과제
            - 중기 (6-12개월) 구조적 변화
            - 장기 협업 문화 구축 방안

            ## 6. 기대 효과
            - 의사결정 속도 개선
            - 업무 효율성 증대
            - 직원 만족도 및 참여도 향상
            - 비즈니스 성과 개선

            회사 지식 베이스에서 관련 정보를 검색하여 현실적이고 실행 가능한 제안을 제시해주세요. 부서 간 협업이 특히 중요한 프로젝트나 업무 흐름을 중심으로 분석해주세요.
            """


# MCP 서버 실행
if __name__ == "__main__":
    print("Dify 지식 검색 MCP 서버를 시작합니다...")
    print(f"API 엔드포인트: {API_ENDPOINT}")
    print(f"지식 베이스 ID: {KNOWLEDGE_ID}")
    mcp.run()