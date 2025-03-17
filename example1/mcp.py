def register_tools(mcp):
    """MCP 서버에 도구 등록"""
    @mcp.tool()
    async def basic_search(
        query: str, 
        top_k: int = 5
    ) -> str:
        """
        기본 RAG를 사용하여 PDF 문서에서 정보 검색
        
        Parameters:
            query: 검색 쿼리 또는 질문
            top_k: 반환할 관련 문서 수
            
        Returns:
            검색 결과 문서 내용
        """
        from example1.rag import search_documents
        return search_documents(query, top_k)