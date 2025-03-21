from typing import List
import os

from dotenv import load_dotenv
from exa_py import Exa
from mcp.server.fastmcp import FastMCP

load_dotenv()

exa_api_key = os.getenv("EXA_API_KEY")
exa = Exa(api_key=exa_api_key)

websearch_config = {
    "parameters": {
        "default_num_results": 5,
        "include_domains": []
    }
}

mcp = FastMCP(
    name="web_search", 
    version="1.0.0",
    description="Web search capability using Exa API that provides real-time internet search results. Supports both basic and advanced search with filtering options including domain restrictions, text inclusion requirements, and date filtering. Returns formatted results with titles, URLs, publication dates, and content summaries."
)

def format_search_results(search_results):
    """
    Converts search results to markdown format.
    
    Args:
        search_results: Exa search results
        
    Returns:
        String in markdown format

    """

    if not search_results.results:
        return "No results found."

    markdown_results = "### Search Results:\n\n"

    for idx, result in enumerate(search_results.results, 1):
        title = result.title if hasattr(result, 'title') and result.title else "No title"
        url = result.url
        published_date = f" (Published: {result.published_date})" if hasattr(result, 'published_date') and result.published_date else ""
        
        markdown_results += f"**{idx}.** [{title}]({url}){published_date}\n"
        
        if hasattr(result, 'summary') and result.summary:
            markdown_results += f"> **Summary:** {result.summary}\n\n"
        else:
            markdown_results += "\n"
    
    return markdown_results
    
@mcp.tool()
async def search_web(query: str, num_results: int = None) -> str:
    """
    Searches the web using Exa API and returns results in markdown format.
    
    Args:
        query: Search query
        num_results: Number of results to return (overrides default setting)
    
    Returns:
        Search results in markdown format

    """

    try:
        search_args = {
            "num_results": num_results or websearch_config["parameters"]["default_num_results"]
        }
        
        search_results = exa.search_and_contents(
            query, 
            summary={"query": "Main points and key takeaways"},
            **search_args
        )
        
        return format_search_results(search_results)
    except Exception as e:
        return f"Error occurred during Exa search: {e}"

@mcp.tool()
async def advanced_search_web(
    query: str, 
    num_results: int = None, 
    include_domains: List[str] = None, 
    include_text: str = None,
    max_age_days: int = None
) -> str:
    """
    Advanced web search using Exa API with additional filtering options.
    
    Args:
        query: Search query
        num_results: Number of results to return (overrides default setting)
        include_domains: List of domains to include in search results
        include_text: Text that must be included in search results
        max_age_days: Maximum age of results in days
        
    Returns:
        Search results in markdown format

    """

    try:
        search_args = {
            "num_results": num_results or websearch_config["parameters"]["default_num_results"]
        }
        
        if include_domains:
            search_args["include_domains"] = include_domains

        elif websearch_config["parameters"]["include_domains"]:
            search_args["include_domains"] = websearch_config["parameters"]["include_domains"]
            
        if include_text:
            search_args["include_text"] = [include_text]
            
        if max_age_days:
            from datetime import datetime, timedelta
            start_date = (datetime.now() - timedelta(days=max_age_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
            search_args["start_published_date"] = start_date
        
        search_results = exa.search_and_contents(
            query, 
            summary={"query": "Main points and key takeaways"},
            **search_args
        )
        
        return format_search_results(search_results)
    except Exception as e:
        return f"Error occurred during Exa advanced search: {e}"

@mcp.resource("help: hantaek@brain-crew.com")
def get_search_help() -> str:
    """Provides help for web search tools."""

    return """
            # Web Search Tool Usage Guide
            
            Provides Claude with real-time web search capability through the Exa API.
            
            ## Basic Search
            The `search_web` tool performs simple web searches.
            - Parameters: 
            - query: Search query
            - num_results: Number of results to return (optional, default: 5)
            
            ## Advanced Search
            The `advanced_search_web` tool provides advanced search with additional filtering options.
            - Parameters:
            - query: Search query
            - num_results: Number of results to return (optional)
            - include_domains: List of domains to include in search results
            - include_text: Text that must be included in search results
            - max_age_days: Maximum age of results in days
            
            ## Usage Examples
            - Basic search: "I'm curious about the latest AI development trends"
            - Advanced search: "Use when you want to search only specific websites or find results containing specific text"

            """

if __name__ == "__main__":
    mcp.run()