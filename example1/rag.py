import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
VECTOR_DIR = Path(__file__).parent / "chroma_db"

def initialize_vectorstore() -> Chroma:
    """벡터 저장소 초기화: 없으면 생성, 있으면 로드"""

    if not DATA_DIR.exists():
        raise ValueError(f"데이터 디렉토리가 없습니다: {DATA_DIR}")
    
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"PDF 파일이 없습니다: {DATA_DIR}")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    if VECTOR_DIR.exists() and any(VECTOR_DIR.iterdir()):
        print(f"기존 벡터 저장소 로드: {VECTOR_DIR}")
        return Chroma(persist_directory=str(VECTOR_DIR), embedding_function=embeddings)
    
    print("새 벡터 저장소 생성 중...")
    
    documents = []
    for pdf_path in pdf_files:
        print(f"PDF 로드 중: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    print(f"생성된 청크 수: {len(chunks)}")
    
    os.makedirs(VECTOR_DIR, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DIR)
    )

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
        vectorstore = initialize_vectorstore()
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        results = retriever.get_relevant_documents(query)
        
        return format_search_results(results)
        
    except Exception as e:
        return f"검색 중 오류가 발생했습니다: {str(e)}"