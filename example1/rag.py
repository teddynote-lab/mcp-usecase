"""
기본 RAG 구현 (LangChain + ChromaDB)

PDF 문서를 로드하고 벡터 저장소에 저장한 후, 쿼리에 대한 관련 문서를 검색합니다.
"""

import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 환경 변수 로드
load_dotenv()

# 기본 경로 설정
DATA_DIR = Path(__file__).parent.parent / "data"
VECTOR_DIR = Path(__file__).parent / "chroma_db"

def initialize_vectorstore() -> Chroma:
    """벡터 저장소 초기화: 없으면 생성, 있으면 로드"""
    # 경로 확인
    if not DATA_DIR.exists():
        raise ValueError(f"데이터 디렉토리가 없습니다: {DATA_DIR}")
    
    # 문서 존재 확인
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"PDF 파일이 없습니다: {DATA_DIR}")
    
    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 벡터 저장소 존재 확인
    if VECTOR_DIR.exists() and any(VECTOR_DIR.iterdir()):
        print(f"기존 벡터 저장소 로드: {VECTOR_DIR}")
        return Chroma(persist_directory=str(VECTOR_DIR), embedding_function=embeddings)
    
    # 벡터 저장소 생성
    print("새 벡터 저장소 생성 중...")
    
    # 문서 로드
    documents = []
    for pdf_path in pdf_files:
        print(f"PDF 로드 중: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())
    
    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"생성된 청크 수: {len(chunks)}")
    
    # 벡터 저장소 생성 및 저장
    os.makedirs(VECTOR_DIR, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DIR)
    )
    # 참고: 최신 버전에서는 persist()가 필요 없음. 자동으로 저장됨
    print(f"벡터 저장소 생성 완료: {VECTOR_DIR}")
    
    return vectorstore

def format_search_results(results: List[Document]) -> str:
    """검색 결과를 가독성 높은 형태로 포맷팅"""
    if not results:
        return "검색 결과가 없습니다."
    
    formatted_text = "# 검색 결과\n\n"
    
    for i, doc in enumerate(results):
        page_num = doc.metadata.get("page", "알 수 없음")
        source = Path(doc.metadata.get("source", "")).name
        
        formatted_text += f"## 결과 {i+1}\n"
        formatted_text += f"**출처:** {source}, 페이지 {page_num}\n\n"
        formatted_text += f"{doc.page_content}\n\n"
        formatted_text += "---\n\n"
    
    return formatted_text

def search_documents(query: str, top_k: int = 5) -> str:
    """사용자 쿼리에 대한 관련 문서 검색"""
    try:
        # 벡터 저장소 초기화
        vectorstore = initialize_vectorstore()
        
        # 유사도 검색
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        results = retriever.get_relevant_documents(query)
        
        # 결과 포맷팅
        return format_search_results(results)
        
    except Exception as e:
        return f"검색 중 오류가 발생했습니다: {str(e)}"