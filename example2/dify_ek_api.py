import logging
import os
import time
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from pathlib import Path

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

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API key setup
API_KEY = "dify-external-knowledge-api-key"
api_key_header = APIKeyHeader(name="Authorization")

# Directory setup
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" 
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# PDF file path (using project shared data folder)
PDF_FILES = list(DATA_DIR.glob("*.pdf"))
PDF_PATH = PDF_FILES[0] if PDF_FILES else DATA_DIR / "sample.pdf"

app = FastAPI(title="Dify External Knowledge API - LangGraph Version")


###### STEP 1. State and Preprocessing Function Definition ######

class KnowledgeState(TypedDict):
    """
    State definition used in LangGraph graph.
    
    Each field represents data passed between graph nodes.

    """

    query: Annotated[str, "User's search query"]

    search_method: Annotated[str, "Search method"]

    top_k: Annotated[int, "Maximum number of results to return"]

    score_threshold: Annotated[float, "Minimum relevance score for inclusion (0.0-1.0)"]

    results: Annotated[List[Dict[str, Any]], "List of search results"]

    vector_db: Annotated[Optional[Any], "Chroma vector DB instance"]

    semantic_retriever: Annotated[Optional[Any], "Semantic search retriever"]
    keyword_retriever: Annotated[Optional[Any], "Keyword-based search retriever"]
    hybrid_retriever: Annotated[Optional[Any], "Hybrid search retriever"]


###### STEP 2. Node Definition ######

class DocumentProcessor:
    """
    Loads PDF files, extracts text, splits into chunks,
    and stores in a vector database (ChromaDB).

    """

    def __init__(self, knowledge_id="test-knowledge-base"):
        self.knowledge_id = knowledge_id
    
    def __call__(self, state: KnowledgeState) -> KnowledgeState:
        """
        Process documents and set up vector storage.

        Args:
            state: Current graph state

        Returns:
            Updated graph state

        """
        
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        
        try:
            embedding = OpenAIEmbeddings(model='text-embedding-3-small')
            chroma_exists = (CHROMA_DB_DIR / "chroma.sqlite3").exists()
            
            if chroma_exists:
                try:
                    vector_db = Chroma(
                        collection_name=self.knowledge_id,
                        embedding_function=embedding,
                        persist_directory=str(CHROMA_DB_DIR)
                    )
                    
                    collection_data = vector_db.get()

                    if not collection_data.get("documents", []):
                        logger.warning("Existing collection is empty. Creating a new one.")
                        raise ValueError("Empty collection")
                    
                except Exception as e:
                    logger.warning(f"Failed to load existing vector store: {str(e)}. Creating a new one.")
                    chroma_exists = False
                    
                    if CHROMA_DB_DIR.exists():
                        backup_dir = f"{CHROMA_DB_DIR}_backup_{int(time.time())}"
                        os.rename(CHROMA_DB_DIR, backup_dir)
                        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
                    
            if not chroma_exists:
                loader = PDFPlumberLoader(str(PDF_PATH))
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=600,
                    chunk_overlap=50
                )
                split_docs = text_splitter.split_documents(docs)
                
                if not split_docs:
                    logger.warning("No text chunks available. Using temporary data.")
                    split_docs = [
                        Document(
                            page_content="This is a test document chunk 1 for Dify external knowledge API.",
                            metadata={
                                "path": str(PDF_PATH),
                                "description": "Test PDF document",
                                "title": PDF_PATH.name
                            }
                        ),
                        Document(
                            page_content="This is a test document chunk 2 about PDF processing and retrieval.",
                            metadata={
                                "path": str(PDF_PATH),
                                "description": "Test PDF document",
                                "title": PDF_PATH.name
                            }
                        ),
                        Document(
                            page_content="This is a test document chunk 3 explaining external knowledge API implementation.",
                            metadata={
                                "path": str(PDF_PATH),
                                "description": "Test PDF document",
                                "title": PDF_PATH.name
                            }
                        )
                    ]
                
                vector_db = Chroma.from_documents(
                    documents=split_docs,
                    embedding=embedding,
                    persist_directory=str(CHROMA_DB_DIR),
                    collection_name=self.knowledge_id
                )
            
            state["vector_db"] = vector_db
            
        except Exception as e:
            logger.error(f"Error during vector store initialization: {str(e)}")
            raise
        
        return state

class RetrieverSetup:
    """
    Sets up semantic, keyword, and hybrid retrievers 
    from the vector database.
    
    """

    def __call__(self, state: KnowledgeState) -> KnowledgeState:
        """
        Configure retrievers.

        Args:
            state: Current graph state

        Returns:
            Updated graph state with configured retrievers

        """
        
        vector_db = state.get("vector_db")

        if vector_db is None:
            logger.error("Vector store not found in state.")
            raise ValueError("Vector store not found in state")
        
        top_k = state.get("top_k", 5)
        
        try:
            semantic_retriever = vector_db.as_retriever(
                search_kwargs={"k": top_k}
            )
            state["semantic_retriever"] = semantic_retriever
            logger.info("Semantic retriever setup complete")
            
            try:
                result = vector_db.get()
                
                if "documents" in result and result["documents"]:
                    docs = result["documents"]
                    metadatas = result.get("metadatas", [None] * len(docs))
                    logger.info(f"Retrieved {len(docs)} documents from ChromaDB.")
                else:
                    logger.warning("Could not retrieve documents from ChromaDB. Creating temporary documents.")
                    docs = ["This is a temporary document for testing purposes."]
                    metadatas = [None]
                
                doc_objects = [
                    Document(
                        page_content=text,
                        metadata=meta if meta else {}
                    )
                    for text, meta in zip(docs, metadatas)
                ]
                
                keyword_retriever = BM25Retriever.from_documents(doc_objects)
                keyword_retriever.k = top_k
                state["keyword_retriever"] = keyword_retriever
                
                hybrid_retriever = EnsembleRetriever(
                    retrievers=[keyword_retriever, semantic_retriever],
                    weights=[0.5, 0.5]
                )
                state["hybrid_retriever"] = hybrid_retriever
                
            except Exception as inner_e:
                logger.error(f"Error during BM25 retriever setup: {str(inner_e)}")
                logger.info("Using semantic retriever only.")
                state["keyword_retriever"] = semantic_retriever  # Fallback
                state["hybrid_retriever"] = semantic_retriever   # Fallback
            
        except Exception as e:
            logger.error(f"Error during retriever setup: {str(e)}")
            raise
        
        return state

class PerformRetrieval:
    """
    Performs search using the appropriate retriever based on user query.

    """

    def __call__(self, state: KnowledgeState) -> KnowledgeState:
        """
        Execute retrieval process.

        Args:
            state: Current graph state

        Returns:
            Updated graph state with search results

        """
        
        query = state.get("query", "")
        search_method = state.get("search_method", "hybrid_search")
        top_k = state.get("top_k", 5)
        score_threshold = state.get("score_threshold", 0.5) 
        logger.info(f"Performing search: query='{query}', method={search_method}, top_k={top_k}")
        
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
            retriever = state.get("hybrid_retriever")
        
        if not retriever:
            logger.error(f"Retriever not found: {search_method}")
            retriever = state.get("hybrid_retriever")
            if not retriever:
                raise ValueError(f"No retriever available in state")
            
            logger.warning(f"Could not find {search_method} retriever, using hybrid retriever instead.")
        
        try:
            docs = retriever.get_relevant_documents(query)  
            docs = docs[:top_k]
            
            results = []
            for i, doc in enumerate(docs):
                metadata = doc.metadata.copy() if hasattr(doc, 'metadata') and doc.metadata else {}
                score = max(0.95 - (i * 0.1), score_threshold)
                
                results.append({
                    "metadata": metadata,
                    "score": score,
                    "title": doc.metadata.get("Title", doc.metadata.get("title", "Document chunk")),
                    "content": doc.page_content
                })
            
            state["results"] = results

            if not results:
                logger.warning("No search results. Adding default response.")

                state["results"] = [{
                    "metadata": {
                        "path": str(PDF_PATH),
                        "description": "Default response"
                    },
                    "score": 0.5,
                    "title": "Default response",
                    "content": f"No relevant documents found for query: '{query}'"
                }]
                
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")

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


###### STEP 3. Graph Creation and Compilation ######

def create_knowledge_graph():
    """
    Creates a LangGraph-based knowledge retrieval graph.
    
    Returns:
        Compiled graph instance
        
    """
    
    graph_builder = StateGraph(KnowledgeState)
    
    graph_builder.add_node("document_processor", DocumentProcessor())
    graph_builder.add_node("retriever_setup", RetrieverSetup())
    graph_builder.add_node("perform_retrieval", PerformRetrieval())
    
    graph_builder.add_edge(START, "document_processor")
    graph_builder.add_edge("document_processor", "retriever_setup")
    graph_builder.add_edge("retriever_setup", "perform_retrieval")
    graph_builder.add_edge("perform_retrieval", END)
    
    return graph_builder.compile()


###### STEP 4. Graph Instance Creation ######

try:
    knowledge_graph = create_knowledge_graph()
    logger.info("Knowledge graph instance creation complete")

except Exception as e:
    logger.error(f"Error creating knowledge graph: {str(e)}")
    knowledge_graph = None


###### STEP 5. API Request and Response Class Definition ######

class RetrievalSetting(BaseModel):
    """Retrieval settings model"""

    top_k: Annotated[int, "Maximum number of results to return"]
    score_threshold: Annotated[float, "Minimum relevance score for inclusion (0.0-1.0)"]


class ExternalKnowledgeRequest(BaseModel):
    """External knowledge API request model"""

    knowledge_id: Annotated[str, "ID of the knowledge base to search"]
    query: Annotated[str, "User search query"]
    search_method: Annotated[str, "Search method (semantic_search, keyword_search, hybrid_search)"] = "hybrid_search"
    retrieval_setting: Annotated[RetrievalSetting, "Retrieval settings"]


###### STEP 6. API Key Validation Function ######

async def verify_api_key(authorization: str = Header(...)):
    """API key validation function"""

    if not authorization.startswith("Bearer "):
        logger.warning("Invalid Authorization header format")

        raise HTTPException(
            status_code=403,
            detail={
                "error_code": 1001,
                "error_msg": "Invalid Authorization header format. Expected 'Bearer ' format."
            }
        )
    
    token = authorization.replace("Bearer ", "")

    if token != API_KEY:
        logger.warning("Authentication failed: Invalid API key")

        raise HTTPException(
            status_code=403,
            detail={
                "error_code": 1002,
                "error_msg": "Authorization failed"
            }
        )
    
    return token


###### STEP 7. API Endpoint Definition ######

@app.post("/retrieval")
async def retrieve_knowledge(
    request: ExternalKnowledgeRequest,
    token: str = Depends(verify_api_key)):
    """Document retrieval API endpoint"""

    logger.info(f"API request received: query='{request.query}'")
    
    if knowledge_graph is None:
        logger.error("Knowledge graph is not initialized.")

        raise HTTPException(status_code=500, detail="Knowledge graph is not initialized")
    
    initial_state = KnowledgeState(
        query=request.query,
        search_method=request.search_method,
        top_k=request.retrieval_setting.top_k,
        score_threshold=request.retrieval_setting.score_threshold,
        results=[],
        vector_db=None,
        semantic_retriever=None,
        keyword_retriever=None,
        hybrid_retriever=None
    )
    
    try:
        final_state = knowledge_graph.invoke(initial_state)
        results = final_state.get("results", [])
        
        response_records = []

        for r in results:
            metadata = r.get("metadata", {})
            if not metadata:
                metadata = {"path": "unknown", "description": ""}

            response_records.append({
                "metadata": metadata,
                "score": r.get("score", 0.5),
                "title": r.get("title", "Document"),
                "content": r.get("content", "No content")
            })
        
        return {"records": response_records}
    
    except Exception as e:
        logger.error(f"Error during knowledge graph execution: {str(e)}")

        return {"records": [{
            "metadata": {
                "path": "error",
                "description": "Error response"
            },
            "score": 0.5,
            "title": "Error",
            "content": f"An error occurred: {str(e)}"
        }]}

@app.get("/health")
async def health_check():
    """Server health check endpoint"""

    health_status = {
        "status": "healthy" if knowledge_graph is not None else "unhealthy",
        "knowledge_graph_initialized": knowledge_graph is not None,
        "openai_api_key_set": os.getenv("OPENAI_API_KEY") is not None,
        "data_directory_exists": DATA_DIR.exists(),
        "chroma_db_directory_exists": CHROMA_DB_DIR.exists(),
        "pdf_exists": PDF_PATH.exists()
    }
    
    return health_status

if __name__ == "__main__":
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)