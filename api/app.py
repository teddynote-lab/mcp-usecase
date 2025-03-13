import logging
import os
import time
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph


###### STEP 0. 기본 설정 ######
# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API 키 설정: 커스터마이즈가 가능합니다.
# Dify 외부 지식 API 설정 시, API Key 값에 해당하는 부분입니다. 
# API Endpoint에 작성하지 않도록 주의해주세요.
API_KEY = "dify-external-knowledge-api-key"
api_key_header = APIKeyHeader(name="Authorization")

# 디렉토리 설정
DATA_DIR = "./data"
CHROMA_DB_DIR = "./chroma_db"

# PDF 파일 경로
PDF_PATH = os.path.join(DATA_DIR, "test.pdf")

# FastAPI 앱 생성
app = FastAPI(title="Dify 외부 지식 API - LangGraph 버전")

# OpenAI API 키 확인
logger.info(f"OPENAI_API_KEY 설정 여부: {'설정됨' if os.getenv('OPENAI_API_KEY') else '설정되지 않음'}")


###### STEP 1. 상태(State) 및 전처리 함수 정의 ######
class KnowledgeState(TypedDict):
    """
    LangGraph 그래프에서 사용되는 상태 정의

    각 필드는 그래프의 노드 간에 전달되는 데이터를 나타냅니다.
    """

    # 입력 쿼리
    query: Annotated[str, "사용자가 입력한 검색 쿼리"]
    # 검색 방법
    search_method: Annotated[str, "검색 방법"]
    # 최대 결과 수
    top_k: Annotated[int, "반환할 최대 결과 수"]
    # 점수 임계값
    score_threshold: Annotated[float, "결과에 포함할 최소 관련성 점수(0.0-1.0)"]
    # 검색 결과
    results: Annotated[List[Dict[str, Any]], "검색 결과 목록"]
    # 벡터 저장소
    vector_db: Annotated[Optional[Any], "Chroma 벡터 DB 인스턴스"]
    # 리트리버
    semantic_retriever: Annotated[Optional[Any], "의미 기반 검색 리트리버"]
    keyword_retriever: Annotated[Optional[Any], "키워드 기반 검색 리트리버"]
    hybrid_retriever: Annotated[Optional[Any], "하이브리드 검색 리트리버"]


###### STEP 2. 노드(Node) 정의 ######
# 1. 문서 처리 및 벡터 저장소 설정 노드
class DocumentProcessor:
    """
    PDF 파일을 로드하고 텍스트를 추출하여 청크로 분할한 후

    벡터 저장소(ChromaDB)에 저장하는 역할을 담당합니다.
    """

    def __init__(self, knowledge_id="test-knowledge-base"):
        """
        DocumnetProcessor 초기화

        Args:
            knowledge_id (str): 지식 베이스 ID
        """

        self.knowledge_id = knowledge_id
    
    def __call__(self, state: KnowledgeState) -> KnowledgeState:
        """
        문서 처리 및 벡터 저장소 설정 실행

        Args:
            state (KnowledgeState): 현재 그래프 상태

        Returns:
            KnowledgeState: 업데이트된 그래프 상태
        """

        logger.info("DocumentProcessor 노드 실행 중...")
        
        # 디렉토리 확인 및 생성
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        
        # PDF 파일 확인
        if not os.path.exists(PDF_PATH):
            logger.warning(f"PDF 파일이 없습니다: {PDF_PATH}. 테스트용 파일을 생성합니다.")
            with open(PDF_PATH, "w") as f:
                f.write("This is a test document for Dify external knowledge API testing. " + 
                        "It contains information about PDF processing and retrieval methods. " +
                        "The document explains how to implement external knowledge API for Dify service.")
        
        try:
            # 임베딩 모델 초기화
            embedding = OpenAIEmbeddings(model='text-embedding-3-small')
            logger.info(f"임베딩 모델 초기화 완료. 모델 객체: {embedding}")
            
            # 기존 ChromaDB 컬렉션이 있는지 확인
            chroma_exists = os.path.exists(os.path.join(CHROMA_DB_DIR, "chroma.sqlite3"))
            
            # 벡터 저장소 초기화
            if chroma_exists:
                logger.info("기존 벡터 저장소 로드 중...")
                try:
                    # 기존 벡터 저장소 로드 시도
                    vector_db = Chroma(
                        collection_name=self.knowledge_id,
                        embedding_function=embedding,
                        persist_directory=CHROMA_DB_DIR
                    )
                    
                    # 데이터가 있는지 확인
                    collection_data = vector_db.get()
                    if not collection_data.get("documents", []):
                        logger.warning("기존 컬렉션이 비어 있습니다. 새로 생성합니다.")
                        raise ValueError("Empty collection")
                        
                except Exception as e:
                    logger.warning(f"기존 벡터 저장소 로드 실패: {str(e)}. 새로 생성합니다.")
                    chroma_exists = False
                    
                    # 기존 DB 디렉토리 백업 후 삭제
                    if os.path.exists(CHROMA_DB_DIR):
                        backup_dir = f"{CHROMA_DB_DIR}_backup_{int(time.time())}"
                        os.rename(CHROMA_DB_DIR, backup_dir)
                        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
                    
            if not chroma_exists:
                logger.info("새 벡터 저장소 생성 중...")
                # PDFPlumberLoader를 사용하여 PDF 로드 (페이지별 Document 객체 반환)
                loader = PDFPlumberLoader(PDF_PATH)
                docs = loader.load()
                logger.info(f"PDF 로드 완료. 페이지 수: {len(docs)}")
                
                # 텍스트 분할
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=600,
                    chunk_overlap=50
                )
                split_docs = text_splitter.split_documents(docs)
                logger.info(f"텍스트 청크 분할 완료. 청크 수: {len(split_docs)}")
                
                if not split_docs:
                    # 텍스트가 없거나 분할 실패한 경우, 테스트 데이터 사용
                    logger.warning("텍스트 청크가 없습니다. 임시 데이터를 사용합니다.")
                    split_docs = [
                        Document(
                            page_content="This is a test document chunk 1 for Dify external knowledge API.",
                            metadata={
                                "path": PDF_PATH,
                                "description": "Test PDF document",
                                "title": os.path.basename(PDF_PATH)
                            }
                        ),
                        Document(
                            page_content="This is a test document chunk 2 about PDF processing and retrieval.",
                            metadata={
                                "path": PDF_PATH,
                                "description": "Test PDF document",
                                "title": os.path.basename(PDF_PATH)
                            }
                        ),
                        Document(
                            page_content="This is a test document chunk 3 explaining external knowledge API implementation.",
                            metadata={
                                "path": PDF_PATH,
                                "description": "Test PDF document",
                                "title": os.path.basename(PDF_PATH)
                            }
                        )
                    ]
                
                # 벡터 저장소 생성
                vector_db = Chroma.from_documents(
                    documents=split_docs,
                    embedding=embedding,
                    persist_directory=CHROMA_DB_DIR,
                    collection_name=self.knowledge_id
                )

                logger.info(f"새 벡터 저장소 생성 완료. 문서 수: {len(split_docs)}")
            
            # 상태에 벡터 저장소 추가
            state["vector_db"] = vector_db
            logger.info("벡터 저장소 상태에 추가 완료")
            
        except Exception as e:
            logger.error(f"벡터 저장소 초기화 중 오류 발생: {str(e)}")
            raise
        
        return state


# 2. 리트리버 설정 노드
class RetrieverSetup:
    """
    리트리버 설정 노드

    벡터 저장소에서 의미 기반, 키워드 기반, 하이브리드 검색을 위한
    리트리버를 설정하는 역할을 담당합니다.
    """

    def __call__(self, state: KnowledgeState) -> KnowledgeState:
        """
        리트리버 설정 실행

        Args:
            state (KnowledgeState): 현재 그래프 상태

        Returns:
            KnowledgeState: 업데이트된 그래프 상태

        Raises:
            ValueError: 벡터 저장소가 상태에 없거나 리트리버 설정 실패 시 발생
        """

        logger.info("RetrieverSetup 노드 실행 중...")
        
        vector_db = state.get("vector_db")
        if vector_db is None:
            logger.error("벡터 저장소가 상태에 없습니다.")
            raise ValueError("Vector store not found in state")
        
        top_k = state.get("top_k", 5)
        
        try:
            # 시맨틱 검색을 위한 리트리버 생성
            semantic_retriever = vector_db.as_retriever(
                search_kwargs={"k": top_k}
            )
            state["semantic_retriever"] = semantic_retriever
            logger.info("시맨틱 리트리버 설정 완료")
            
            # BM25 리트리버용 문서 가져오기
            try:
                # 먼저 Chroma에서 문서 가져오기 시도
                result = vector_db.get()
                logger.info(f"ChromaDB get() 결과 키: {list(result.keys())}")
                
                if "documents" in result and result["documents"]:
                    docs = result["documents"]
                    logger.info(f"ChromaDB에서 {len(docs)} 개의 문서를 가져왔습니다.")
                else:
                    # 문서가 없는 경우, 임시 문서 생성
                    logger.warning("ChromaDB에서 문서를 가져올 수 없습니다. 임시 문서를 생성합니다.")
                    docs = ["This is a temporary document for testing purposes."]
                
                doc_objects = [
                    Document(page_content=text)
                    for text in docs
                ]
                
                # 키워드 기반 검색을 위한 BM25 리트리버 생성
                keyword_retriever = BM25Retriever.from_documents(doc_objects)
                keyword_retriever.k = top_k
                state["keyword_retriever"] = keyword_retriever
                logger.info("키워드 리트리버 설정 완료")
                
                # 하이브리드 검색 리트리버 생성
                hybrid_retriever = EnsembleRetriever(
                    retrievers=[keyword_retriever, semantic_retriever],
                    weights=[0.5, 0.5]
                )
                state["hybrid_retriever"] = hybrid_retriever
                logger.info("하이브리드 리트리버 설정 완료")
                
            except Exception as inner_e:
                # BM25 리트리버 설정 실패 시 시맨틱 리트리버만 사용
                logger.error(f"BM25 리트리버 설정 중 오류 발생: {str(inner_e)}")
                logger.info("시맨틱 리트리버만 사용합니다.")
                state["keyword_retriever"] = semantic_retriever  # 대체용
                state["hybrid_retriever"] = semantic_retriever   # 대체용
            
        except Exception as e:
            logger.error(f"리트리버 설정 중 오류 발생: {str(e)}")
            raise
        
        return state


# 3. 검색 수행 노드
class PerformRetrieval:
    """
    검색 수행 노드

    사용자 쿼리에 대해 적절한 리트리버를 사용하여
    관련 문서를 검색하는 역할을 담당합니다.
    """

    def __call__(self, state: KnowledgeState) -> KnowledgeState:
        """
        검색 수행 실행

        Args:
            state (KnowledgeState): 현재 그래프 상태

        Returns:
            KnowledgeState: 업데이트 된 그래프 상태
        """

        logger.info("PerformRetrieval 노드 실행 중...")
        
        query = state.get("query", "")
        search_method = state.get("search_method", "hybrid_search")
        top_k = state.get("top_k", 5)
        score_threshold = state.get("score_threshold", 0.5)
        
        logger.info(f"검색 수행: 쿼리='{query}', 검색 방법={search_method}, top_k={top_k}")
        
        # 리트리버 선택
        retriever = None
        if search_method == "keyword_search":
            retriever = state.get("keyword_retriever")
        elif search_method == "semantic_search":
            retriever = state.get("semantic_retriever")
        elif search_method == "hybrid_search":
            retriever = state.get("hybrid_retriever")
        elif search_method == "full_text_search":
            retriever = state.get("keyword_retriever")
        else:
            retriever = state.get("hybrid_retriever")  # 기본값
        
        if not retriever:
            logger.error(f"리트리버를 찾을 수 없습니다: {search_method}")
            # 대체 리트리버 사용
            retriever = state.get("semantic_retriever")
            if not retriever:
                raise ValueError(f"No retriever available in state")
            
            logger.warning(f"{search_method} 리트리버를 찾을 수 없어 시맨틱 리트리버로 대체합니다.")
        
        try:
            # 검색 수행
            docs = retriever.get_relevant_documents(query)
            logger.info(f"검색 결과: {len(docs)}개 문서 검색됨")
            
            # 결과 제한
            docs = docs[:top_k]
            
            # 결과 형식 변환
            results = []
            for i, doc in enumerate(docs):

                ## [메타데이터 전체 저장] - 원본 메타데이터를 그대로 사용
                metadata = doc.metadata.copy() if hasattr(doc, 'metadata') and doc.metadata else {}

                # 간단한 점수 계산
                score = max(0.95 - (i * 0.1), score_threshold)
                
                results.append({
                    "metadata": metadata,
                    "score": score,
                    "title": doc.metadata.get("Title", doc.metadata.get("title", "Document chunk")),
                    "content": doc.page_content
                })
            
            # 상태에 결과 추가
            state["results"] = results
            logger.info(f"결과 처리 완료. {len(results)}개 결과")
            
            # 결과가 없는 경우 기본 응답 추가
            if not results:
                logger.warning("검색 결과가 없습니다. 기본 응답을 추가합니다.")
                state["results"] = [{
                    "metadata": {
                        "path": PDF_PATH,
                        "description": "Default response"
                    },
                    "score": 0.5,
                    "title": "Default response",
                    "content": f"No relevant documents found for query: '{query}'"
                }]
                
        except Exception as e:
            logger.error(f"검색 수행 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본 응답
            state["results"] = [{
                "metadata": {
                    "path": "error",
                    "description": "Error occurred during retrieval"
                },
                "score": 0.5,
                "title": "Error",
                "content": f"An error occurred during retrieval: {str(e)}"
            }]
        
        return state


###### STEP 3. 그래프 생성, 노드 추가, 엣지 추가, 컴파일 함수 정의 ######
def create_knowledge_graph():
    """
    LangGraph 기반 지식 검색 그래프 생성

    그래프 생성, 노드 추가, 노드 간 연결, 그래프 컴파일을 수행합니다.

    Returns:
        StateGraph: 컴파일된 그래프 인스턴스
    """

    logger.info("지식 그래프 생성 중...")
    
    # 그래프 생성
    graph_builder = StateGraph(KnowledgeState)
    
    # 노드 추가
    graph_builder.add_node("document_processor", DocumentProcessor())
    graph_builder.add_node("retriever_setup", RetrieverSetup())
    graph_builder.add_node("perform_retrieval", PerformRetrieval())
    
    # 노드 간 연결
    graph_builder.add_edge(START, "document_processor")
    graph_builder.add_edge("document_processor", "retriever_setup")
    graph_builder.add_edge("retriever_setup", "perform_retrieval")
    graph_builder.add_edge("perform_retrieval", END)
    
    # 그래프 컴파일
    logger.info("그래프 컴파일 완료")
    return graph_builder.compile()


###### STEP 4. 그래프 인스턴스 생성 ######
try:
    knowledge_graph = create_knowledge_graph()
    logger.info("지식 그래프 인스턴스 생성 완료")
except Exception as e:
    logger.error(f"지식 그래프 생성 중 오류 발생: {str(e)}")
    knowledge_graph = None


###### STEP 5. API 요청 및 응답과 관련한 클래스 정의 ######
class RetrievalSetting(BaseModel):
    """
    검색 설정 모델

    API 요청 시 사용되는 검색 관련 파라미터를 정의합니다.
    """

    top_k: Annotated[int, "반환할 최대 결과 수"]
    score_threshold: Annotated[float, "결과에 포함할 최소 관련성 점수 (0.0-1.0)"]


class ExternalKnowledgeRequest(BaseModel):
    """
    외부 지식 API 요청 모델

    API 요청 본문의 구조를 정의합니다.
    """

    knowledge_id: Annotated[str, "검색할 지식 베이스의 ID"]
    query: Annotated[str, "사용자의 검색 쿼리"]
    retrieval_setting: Annotated[RetrievalSetting, "검색 설정"]


###### STEP 6. API 키 검증 함수 정의 ######
async def verify_api_key(authorization: str = Header(...)):
    """
    API 키 검증 함수

    요청 헤더에서 API 키를 추출하고 유효성을 검증합니다.

    Args:
        authorization (str): Authorization 헤더 값

    Returns:
        str: 유효한 API 키

    Raises:
        HTTPException: 인증 실패 시 발생
    """

    if not authorization.startswith("Bearer "):
        logger.warning("올바르지 않은 Authorization 헤더 형식")
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": 1001,
                "error_msg": "Invalid Authorization header format. Expected 'Bearer ' format."
            }
        )
    token = authorization.replace("Bearer ", "")
    if token != API_KEY:
        logger.warning("인증 실패: 유효하지 않은 API 키")
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": 1002,
                "error_msg": "Authorization failed"
            }
        )
    return token


###### STEP 7. 그래프 실행과 API 엔드포인트 연결 ######
# API 엔드포인트 정의
# Dify 워크플로우와 Retrieval 테스트에서는 해당 엔드포인트로 요청
@app.post("/retrieval")
async def retrieve_knowledge(
    request: ExternalKnowledgeRequest,
    token: str = Depends(verify_api_key)):
    """
    문서 검색 API 엔드포인트

    사용자 쿼리를 기반으로 관련 문서를 검색하고 결과를 반환합니다.
    설정된 API 키(dify-external-knowledge-api-key)로만 접근 가능합니다.

    Args:
        request (ExternalKnowledgeRequest): API 요청 본문
        token (str): 인증 토큰

    Returns:
        dict: 검색 결과가 포함된 응답

    Raises:
        HTTPException: 그래프 실행 오류 시 발생
    """

    logger.info(f"API 요청 수신: query='{request.query}'")
    
    if knowledge_graph is None:
        logger.error("지식 그래프가 초기화되지 않았습니다.")
        raise HTTPException(status_code=500, detail="Knowledge graph is not initialized")
    
    # 초기 상태 설정
    initial_state = KnowledgeState(
        query=request.query,
        search_method="semantic_search", # UI에서 설정할 수 없기 때문에 현재 적힌 값으로 적용
        top_k=request.retrieval_setting.top_k,
        score_threshold=request.retrieval_setting.score_threshold,
        results=[],
        vector_db=None,
        semantic_retriever=None,
        keyword_retriever=None,
        hybrid_retriever=None
    )
    
    try:
        # 그래프 실행
        logger.info("지식 그래프 실행 중...")
        final_state = knowledge_graph.invoke(initial_state)
        logger.info("지식 그래프 실행 완료")
        
        # 결과 추출 - results 키 사용
        results = final_state.get("results", [])
        logger.info(f"추출된 결과: {len(results)}개")
        
        # 응답 생성
        response_records = []
        for r in results:
            # 메타데이터 전체를 추출
            metadata = r.get("metadata", {})

            # 메타데이터가 없으면 기본값으로 설정
            if not metadata:
                metadata = {
                    "path": "unknown",
                    "description": ""
                }

            response_records.append({
                "metadata": metadata,
                "score": r.get("score", 0.5),
                "title": r.get("title", "Document"),
                "content": r.get("content", "No content")
            })
        
        # 응답 반환 - 명시적 딕셔너리 생성
        return {"records": response_records}
    
    except Exception as e:
        logger.error(f"지식 그래프 실행 중 오류 발생: {str(e)}")
        # 오류 발생 시 기본 응답 반환
        return {"records": [{
            "metadata": {
                "path": "error",
                "description": "Error response"
            },
            "score": 0.5,
            "title": "Error",
            "content": f"An error occurred: {str(e)}"
        }]}


# 상태 확인 엔드포인트
@app.get("/health")
async def health_check():
    """
    서버 상태 확인 엔드포인트
    
    API 서버 및 관련 구성 요소의 상태를 확인하고 반환합니다.

    Returns:
        dict: 서버 상태 정보
    """
    
    health_status = {
        "status": "healthy" if knowledge_graph is not None else "unhealthy",
        "knowledge_graph_initialized": knowledge_graph is not None,
        "openai_api_key_set": os.getenv("OPENAI_API_KEY") is not None,
        "data_directory_exists": os.path.exists(DATA_DIR),
        "chroma_db_directory_exists": os.path.exists(CHROMA_DB_DIR),
        "pdf_exists": os.path.exists(PDF_PATH)
    }
    return health_status

if __name__ == "__main__":
    logger.info("서버 시작 중...")
    uvicorn.run(app, host="0.0.0.0", port=8000)