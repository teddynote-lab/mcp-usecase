import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from mcp.server.fastmcp import FastMCP

from rag.pdf import PDFRetrievalChain
import config

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", config.DATA_DIR))
pdf_files = list(DATA_DIR.glob("*.pdf"))
pdf_paths = [str(path) for path in pdf_files]

VECTOR_DIR = Path(os.getenv("VECTOR_DIR", config.VECTOR_DIR))

rag_chain = PDFRetrievalChain(
    source_uri = pdf_paths,
    persist_directory = str(VECTOR_DIR),
    k = config.DEFAULT_TOP_K,
    embedding_model = config.DEFAULT_EMBEDDING_MODEL,
    llm_model = config.DEFAULT_LLM_MODEL
).initialize()

mcp = FastMCP(
    name="RAG",
    version="0.0.1",
    description="RAG Search(keyword, semantic, hybrid)"
)

def format_search_results(docs: List[Document]) -> str:
    """
    Format search results as markdown.
    
    Args:
        docs: List of documents to format
        
    Returns:
        Markdown formatted search results
    """
    if not docs:
        return "No relevant information found."
    
    markdown_results = "## Search Results\n\n"
    
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", None)
        page_info = f" (Page: {page+1})" if page is not None else ""
        
        markdown_results += f"### Result {i}{page_info}\n\n"
        markdown_results += f"{doc.page_content}\n\n"
        markdown_results += f"Source: {source}\n\n"
        markdown_results += "---\n\n"
    
    return markdown_results

@mcp.tool()
async def keyword_search(query: str, top_k: int = 5) -> str:
    """
    Perform keyword-based search on documents.
    
    Args:
        query: Search query
        top_k: Number of results to return
        
    Returns:
        Markdown formatted search results
    """
    try:
        results = rag_chain.search_keyword(query, top_k)
        return format_search_results(results)
    except Exception as e:
        return f"An error occurred during search: {str(e)}"

@mcp.tool()
async def semantic_search(query: str, top_k: int = 5) -> str:
    """
    Perform semantic search on documents.
    
    Args:
        query: Search query
        top_k: Number of results to return
        
    Returns:
        Markdown formatted search results
    """
    try:
        results = rag_chain.search_semantic(query, top_k)
        return format_search_results(results)
    except Exception as e:
        return f"An error occurred during search: {str(e)}"

@mcp.tool()
async def hybrid_search(query: str, top_k: int = 5) -> str:
    """
    Perform hybrid search (keyword + semantic) on documents.
    
    Args:
        query: Search query
        top_k: Number of results to return
        
    Returns:
        Markdown formatted search results
    """
    try:
        results = rag_chain.search_hybrid(query, top_k)
        return format_search_results(results)
    except Exception as e:
        return f"An error occurred during search: {str(e)}"

if __name__ == "__main__":
    mcp.run()