import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from mcp.server.fastmcp import FastMCP

from rag import PDFRetrievalChain
import config

load_dotenv()

# 데이터 디렉토리 설정
DATA_DIR = Path(os.getenv("DATA_DIR", config.DATA_DIR))
pdf_files = list(DATA_DIR.glob("*.pdf"))
pdf_paths = [str(path) for path in pdf_files]

# 벡터 디렉토리 설정
VECTOR_DIR = Path(os.getenv("VECTOR_DIR", config.VECTOR_DIR))

# PDF 검색 체인 초기화
rag_chain = PDFRetrievalChain(
    source_uri = pdf_paths,
    persist_directory = str(VECTOR_DIR),
    k = config.DEFAULT_TOP_K,
    embedding_model = config.DEFAULT_EMBEDDING_MODEL,
    llm_model = config.DEFAULT_LLM_MODEL
).initialize()

# FastMCP 인스턴스 생성
mcp = FastMCP(
    name="RAG",
    version="0.0.1",
    description="RAG Search(keyword, semantic, hybrid)"
)

# 검색 결과를 마크다운 형식으로 포맷팅
def format_search_results(docs: List[Document]) -> str:
    """
    검색 결과를 마크다운 형식으로 포맷팅합니다.
    
    Args:
        docs: 포맷팅할 문서 목록
        
    Returns:
        마크다운 형식의 검색 결과

    """

    if not docs:
        return "관련 정보가 없습니다."
    
    markdown_results = "## 검색 결과\n\n"
    
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "알 수 없는 출처")
        page = doc.metadata.get("page", None)
        page_info = f" (페이지: {page+1})" if page is not None else ""
        
        markdown_results += f"### 결과 {i}{page_info}\n\n"
        markdown_results += f"{doc.page_content}\n\n"
        markdown_results += f"출처: {source}\n\n"
        markdown_results += "---\n\n"
    
    return markdown_results

# 키워드 기반 검색
@mcp.tool()
async def keyword_search(query: str, top_k: int = 5) -> str:
    """
    PDF 문서에서 키워드 기반 검색을 수행합니다.
    정확한 단어/구문 일치를 기반으로 가장 관련성 높은 결과를 반환합니다.
    특정 용어, 정의 또는 정확한 구문을 찾기에 이상적입니다.
    
    Parameters:
        query: 검색 쿼리
        top_k: 반환할 결과 수

    """

    try:
        results = rag_chain.search_keyword(query, top_k)
        return format_search_results(results)
    except Exception as e:
        return f"검색 중 오류가 발생했습니다: {str(e)}"

# 의미 기반 검색
@mcp.tool()
async def semantic_search(query: str, top_k: int = 5) -> str:
    """
    PDF 문서에서 의미 기반 검색을 수행합니다.
    쿼리와 의미적으로 유사한 콘텐츠를 찾아 정확한 단어 일치 없이도 관련 정보를 제공합니다.
    개념적 질문, 주제 이해 또는 주제와 관련된 정보가 필요할 때 가장 적합합니다.
    
    Parameters:
        query: 검색 쿼리
        top_k: 반환할 결과 수

    """

    try:
        results = rag_chain.search_semantic(query, top_k)
        return format_search_results(results)
    except Exception as e:
        return f"검색 중 오류가 발생했습니다: {str(e)}"

# "description": "\nPerforms hybrid search (keyword + semantic) on PDF documents.\n
# Combines exact keyword matching and semantic similarity to deliver optimal results.\n
# The most versatile search option for general questions or when unsure which search type is best.\n\n
# Parameters:\n
#     query: Search query\n
#     top_k: Number of results to return
# \n\n"



## 하이브리드 검색 (키워드 + 의미)
@mcp.tool()
async def hybrid_search(query: str, top_k: int = 5) -> str:
    """
    PDF 문서에서 하이브리드 검색(키워드 + 의미)을 수행합니다.
    정확한 키워드 매칭과 의미적 유사성을 결합하여 최적의 결과를 제공합니다.
    일반적인 질문이나 어떤 검색 유형이 가장 적합한지 확신이 서지 않을 때 가장 다재다능한 검색 옵션입니다.
    
    Parameters:
        query: 검색 쿼리
        top_k: 반환할 결과 수

    """

    try:
        results = rag_chain.search_hybrid(query, top_k)
        return format_search_results(results)
    except Exception as e:
        return f"검색 중 오류가 발생했습니다: {str(e)}"

# MCP 서버 실행
if __name__ == "__main__":
    mcp.run()