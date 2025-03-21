from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings

class RetrievalChain(ABC):
    """
    Abstract base class for RAG search implementations.
    
    This class provides a template for different document retrieval chains,
    allowing for customization of document loading, splitting, vectorization,
    and various search methods.
    """
    
    def __init__(self, **kwargs) -> None:
        """
        Initialize a RetrievalChain with configuration parameters.
        
        Args:
            **kwargs: Keyword arguments including:
                source_uri: Paths to source documents
                k: Number of results to return (default: 5)
                embedding_model: Model name for embeddings (default: OpenAI "text-embedding-3-small")
                persist_directory: Directory to persist vector store
        """

        self.source_uri = kwargs.get("source_uri", [])
        self.k = kwargs.get("k", 5)
        self.embedding_model = kwargs.get("embedding_model", "text-embedding-3-small")
        self.persist_directory = kwargs.get("persist_directory", None)
        self.embeddings = None
        self.vectorstore = None
        self.retrievers = None
        self.split_docs = None
    
    @abstractmethod
    def load_documents(self, source_uris: List[str]) -> List[Document]:
        """
        Load documents from source URIs.
        
        Args:
            source_uris: List of file paths or URIs to load documents from
            
        Returns:
            List of loaded documents
        """

        pass
    
    @abstractmethod
    def create_text_splitter(self) -> Any:
        """
        Create a text splitter appropriate for the document type.
        
        Returns:
            A text splitter instance
        """

        pass
    
    def split_documents(self, docs: List[Document], text_splitter: Any) -> List[Document]:
        """
        Split documents into chunks using the provided text splitter.
        
        Args:
            docs: Documents to split
            text_splitter: Text splitter instance
            
        Returns:
            Split document chunks
        """

        return text_splitter.split_documents(docs)
    
    def create_embedding(self) -> Any:
        """
        Create an embedding model instance.
        
        Returns:
            An embeddings model instance
        """

        return OpenAIEmbeddings(model=self.embedding_model)
    
    @abstractmethod
    def create_vectorstore(self, split_docs: List[Document]) -> Any:
        """
        Create a vector store from split documents.
        
        Args:
            split_docs: Split document chunks
            
        Returns:
            A vector store instance
        """

        pass
    
    def create_semantic_retriever(self, vectorstore: Any) -> BaseRetriever:
        """
        Create a semantic search retriever.
        
        Args:
            vectorstore: Vector store instance
            
        Returns:
            A semantic search retriever
        """

        return vectorstore.as_retriever(
            search_kwargs={"k": self.k}
        )
    
    def create_keyword_retriever(self, split_docs: List[Document]) -> BaseRetriever:
        """
        Create a keyword-based search retriever.
        
        Args:
            split_docs: Split document chunks
            
        Returns:
            A keyword search retriever
        """

        return BM25Retriever.from_documents(split_docs, k=self.k)
    
    def create_hybrid_retriever(self, split_docs: List[Document], vectorstore: Any) -> BaseRetriever:
        """
        Create a hybrid search retriever combining keyword and semantic search.
        
        Args:
            split_docs: Split document chunks
            vectorstore: Vector store instance
            
        Returns:
            A hybrid search retriever
        """

        bm25_retriever = self.create_keyword_retriever(split_docs)
        dense_retriever = self.create_semantic_retriever(vectorstore)
        
        return EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.5, 0.5]
        )
    
    def create_retrievers(self, split_docs: List[Document]) -> Dict[str, BaseRetriever]:
        """
        Create all retriever types.
        
        Args:
            split_docs: Split document chunks
            
        Returns:
            Dictionary of retrievers by search type
        """

        self.embeddings = self.create_embedding()
        self.vectorstore = self.create_vectorstore(split_docs)
        
        return {
            "semantic": self.create_semantic_retriever(self.vectorstore),
            "keyword": self.create_keyword_retriever(split_docs),
            "hybrid": self.create_hybrid_retriever(split_docs, self.vectorstore)
        }
    
    def initialize(self) -> "RetrievalChain":
        """
        Initialize the retrieval chain by loading documents, splitting them,
        and creating retriever instances.
        
        Returns:
            The initialized retrieval chain instance
        """

        docs = self.load_documents(self.source_uri)
        if not docs:
            print("No documents were loaded.")
            return self
        
        text_splitter = self.create_text_splitter()
        self.split_docs = self.split_documents(docs, text_splitter)
        
        self.retrievers = self.create_retrievers(self.split_docs)
        
        print(f"Initialization complete: {len(self.split_docs)} chunks created")
        return self
    
    def search_semantic(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Perform semantic search on the loaded documents.
        
        Args:
            query: Search query
            k: Number of results to return, overrides self.k
            
        Returns:
            Relevant documents
            
        Raises:
            ValueError: If the retrieval chain is not initialized
        """

        if not hasattr(self, 'retrievers') or self.retrievers is None:
            raise ValueError("Initialization required. Call initialize() method first.")
        
        k = k or self.k
        retriever = self.retrievers["semantic"]
        retriever.search_kwargs["k"] = k
        
        return retriever.get_relevant_documents(query)
    
    def search_keyword(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Perform keyword-based search on the loaded documents.
        
        Args:
            query: Search query
            k: Number of results to return (Note: BM25Retriever may not support dynamic k)
            
        Returns:
            Relevant documents
            
        Raises:
            ValueError: If the retrieval chain is not initialized
        """

        if not hasattr(self, 'retrievers') or self.retrievers is None:
            raise ValueError("Initialization required. Call initialize() method first.")
        
        return self.retrievers["keyword"].get_relevant_documents(query)
    
    def search_hybrid(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Perform hybrid search (keyword + semantic) on the loaded documents.
        
        Args:
            query: Search query
            k: Number of results to return (Note: EnsembleRetriever may not support dynamic k)
            
        Returns:
            Relevant documents
            
        Raises:
            ValueError: If the retrieval chain is not initialized
        """

        if not hasattr(self, 'retrievers') or self.retrievers is None:
            raise ValueError("Initialization required. Call initialize() method first.")
        
        return self.retrievers["hybrid"].get_relevant_documents(query)
    
    def search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Default search method that uses semantic search.
        
        Args:
            query: Search query
            k: Number of results to return, overrides self.k
            
        Returns:
            Relevant documents
        """
        
        return self.search_semantic(query, k)
