from typing import List, Optional, Any
import os

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

from rag.base import RetrievalChain

class PDFRetrievalChain(RetrievalChain):
    """
    PDF-specific implementation of the RetrievalChain.
    
    This class specializes in loading, splitting, and indexing PDF documents
    for retrieval.
    """
    
    def __init__(self, 
                 source_uri: List[str], 
                 persist_directory: Optional[str] = None,
                 **kwargs) -> None:
        """
        Initialize a PDF retrieval chain.
        
        Args:
            source_uri: List of PDF file paths
            persist_directory: Directory to persist vector store
            **kwargs: Additional keyword arguments for the base RetrievalChain
        """

        super().__init__(source_uri=source_uri, persist_directory=persist_directory, **kwargs)
    
    def load_documents(self, source_uris: List[str]) -> List[Document]:
        """
        Load PDF documents from file paths.
        
        Args:
            source_uris: List of PDF file paths
            
        Returns:
            List of loaded documents
        """

        docs = []
        for source_uri in source_uris:
            if not os.path.exists(source_uri):
                print(f"File not found: {source_uri}")
                continue
                
            print(f"Loading PDF: {source_uri}")
            loader = PDFPlumberLoader(source_uri)
            docs.extend(loader.load())
        
        return docs
    
    def create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """
        Create a text splitter optimized for PDF documents.
        
        Returns:
            A text splitter instance suitable for PDFs
        """
        
        return RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=50
        )
    
    def create_vectorstore(self, split_docs: List[Document]) -> Any:
        """
        Create a vector store from split PDF documents.
        
        Args:
            split_docs: Split document chunks
            
        Returns:
            A vector store instance
            
        Raises:
            ValueError: If there are no split documents
        """
        
        if not split_docs:
            raise ValueError("No split documents available.")
            
        if self.persist_directory:
            os.makedirs(self.persist_directory, exist_ok=True)
            
            if os.path.exists(self.persist_directory) and any(os.listdir(self.persist_directory)):
                print(f"Loading existing vector store: {self.persist_directory}")

                return Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.create_embedding()
                )
        
        print("Creating new vector store...")

        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.create_embedding(),
            persist_directory=self.persist_directory
        )
        
        return vectorstore