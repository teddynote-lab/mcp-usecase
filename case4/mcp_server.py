import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key = tavily_api_key)

websearch_config = {
    "parameters": {
        "default_num_results": 5,
        "include_domains": []
    }
}

mcp = FastMCP(
    name="web_search", 
    version="1.0.0",
    description="Web search capability using Tavily API that provides real-time internet search results. Supports both basic and advanced search with filtering options including domain restrictions, text inclusion requirements, and date filtering. Returns formatted results with titles, URLs, publication dates, and content summaries."
)

def format_search_results(response):
    """
    Converts search results to markdown format.
    
    Args:
        search_results: Exa search results
        
    Returns:
        String in markdown format

    """

    if not response.get("results"):
        return "No results found."

    markdown_results = "### Search Results:\n\n"

    for idx, result in enumerate(response.get("results", []), 1):
        title = result.get("title", "No title")
        url = result.get("url", "")
        published_date = result.get("published_date", "")
        content = result.get("content", "")
        score = result.get("score", 0)
        
        date_info = f" (Published: {published_date})" if published_date else ""
        
        markdown_results += f"**{idx}.** [{title}]({url}){date_info}\n"
        markdown_results += f"**Relevance Score:** {score:.2f}\n"
        
        if content:
            markdown_results += f"> **Content:** {content}\n\n"
        else:
            markdown_results += "\n"
    
    if response.get("answer"):
        markdown_results += f"\n### Answer:\n{response.get('answer')}\n\n"
    
    if response.get("response_time"):
        markdown_results += f"\n*Search completed in {response.get('response_time'):.2f} seconds*"
    
    return markdown_results
    
@mcp.tool()
async def search_web(query: str, num_results: int = None) -> str:
    """
    Performs real-time web search using the Tavily API.
    Returns latest search results in markdown format including titles, URLs, and content summaries.
    Use when you need current information, recent events, or data not available in your training.

    
    Parameters:
        query: Search query
        num_results: Number of results to return (default: 5)

    """

    try:
        search_args = {
            "max_results": num_results or websearch_config["parameters"]["default_num_results"],
            "search_depth": "basic"
        }
        
        search_results = tavily_client.search(
            query=query,
            **search_args
        )
        
        return format_search_results(search_results)
    except Exception as e:
        return f"Error occurred during Tavily search: {e}"

@mcp.resource("help: dev@brain-crew.com")
def get_search_help() -> str:
    """Provides help for web search tools."""

    return """
            # Web Search Tool Usage Guide
            
            Provides Claude with real-time web search capability through the Tavily API.
            
            ## Web Search
            The `search_web` tool performs simple web searches.
            - Parameters: 
            - query: Search query
            - num_results: Number of results to return (optional, default: 5)
            
            ## Examples
            - Web search: "I'm curious about the latest AI development trends"

            """

if __name__ == "__main__":
    mcp.run()