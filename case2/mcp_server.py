import httpx
import os
import json
from typing import Dict

from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

API_ENDPOINT = os.getenv("DIFY_API_ENDPOINT", "http://localhost:8000/retrieval")
API_KEY = os.getenv("DIFY_API_KEY", "dify-external-knowledge-api-key")
KNOWLEDGE_ID = os.getenv("DIFY_KNOWLEDGE_ID", "test-knowledge-base")

mcp = FastMCP(
    name="Dify External Knowledge API",
    version="0.0.1",
    description="Three search methods(semantic_search, keyword_search, hybrid_search) using the Dify External Knowledge API specification"
)

def format_search_results(data: Dict) -> str:
    """Formats search results in a readable form."""

    records = data.get("records", [])
    
    if not records:
        return "No search results found."
    
    formatted_results = "# Search Results\n\n"
    
    for i, record in enumerate(records):
        content = record.get("content", "")
        score = record.get("score", 0)
        title = record.get("title", f"Result {i+1}")
        metadata = record.get("metadata", {})
        
        # Extract metadata if available
        source_info = []
        if "title" in metadata:
            source_info.append(f"File: {os.path.basename(metadata['title'])}")
        elif "path" in metadata:
            source_info.append(f"File: {os.path.basename(metadata['path'])}")
        if "page" in metadata:
            source_info.append(f"Page: {metadata['page']}")
            
        source_text = " | ".join(source_info) if source_info else "No source information"
        
        formatted_results += f"## {title} (Relevance: {score:.2f})\n"
        formatted_results += f"{source_text}\n\n"
        formatted_results += f"{content}\n\n"
        formatted_results += "---\n\n"
    
    formatted_results += "This information was retrieved via the Dify External Knowledge API."
    return formatted_results

@mcp.tool()
async def dify_ek_search(
    query: str, 
    top_k: int = 5, 
    score_threshold: float = 0.5,
    search_method: str = "hybrid_search",
    ctx = None
) -> str:
    """
    Searches for information in Dify External knowledge base.
    Returns search results with document content, relevance scores, and source information.
    Use when you need to find specific information in enterprise documents, knowledge bases, or specialized content.
    
    Parameters:
        query: Search question or keywords
        top_k: Maximum number of results to return
        score_threshold: Minimum relevance score for inclusion (0.0-1.0)
        search_method: Search method (semantic_search, keyword_search, hybrid_search)

    """

    if ctx:
        ctx.info(f"Search query: {query}")
        ctx.info(f"Maximum results: {top_k}")
        ctx.info(f"Minimum score: {score_threshold}")
    
    # Input validation
    if not query or not query.strip():
        return "Error: Search query is empty."
    
    if top_k < 1:
        top_k = 1
    elif top_k > 20:
        top_k = 20
    
    if score_threshold < 0:
        score_threshold = 0
    elif score_threshold > 1:
        score_threshold = 1
    
    try:
        if ctx:
            ctx.info(f"Calling Dify API: {API_ENDPOINT}")
        
        request_data = {
            "knowledge_id": KNOWLEDGE_ID,
            "query": query,
            "search_method": search_method,
            "retrieval_setting": {
                "top_k": top_k,
                "score_threshold": score_threshold
            }
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                API_ENDPOINT,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}"
                },
                json=request_data
            )
            
            if response.status_code != 200:
                error_message = f"Dify API error: HTTP {response.status_code}"
                try:
                    error_detail = response.json()
                    if isinstance(error_detail, dict) and "error_msg" in error_detail:
                        error_message += f" - {error_detail['error_msg']}"
                except:
                    error_message += f" - {response.text[:100]}"
                
                if ctx:
                    ctx.error(error_message)
                return f"Search failed\n\n{error_message}"
            
            try:
                data = response.json()
                return format_search_results(data)
                
            except json.JSONDecodeError:
                if ctx:
                    ctx.error("JSON parsing error")
                return "Search failed\n\nCould not parse API response."
            
    except httpx.RequestError as e:
        error_message = f"API request error: {str(e)}"
        if ctx:
            ctx.error(error_message)
        return f"Search failed\n\n{error_message}"
    
    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        if ctx:
            ctx.error(error_message)
        return f"Search failed\n\n{error_message}"

@mcp.prompt()
def ai_trend_learning_guide(
    topic: str = "",
    learning_level: str = "beginner",
    time_horizon: str = "short-term"
) -> str:
    """
    Creates customized AI learning guides based on SPRI monthly AI reports.
    Provides latest AI trends and learning roadmaps tailored to learner level and desired timeframe.
    Perfect for educational planning, career development in AI, or understanding current AI landscape.
    
    Parameters:
        topic: AI topic of interest (optional - e.g., "generative AI", "computer vision", "NLP")
        learning_level: Learner level ("beginner", "intermediate", "advanced")
        time_horizon: Learning plan duration ("short-term", "medium-term", "long-term")

    """
    
    level_approaches = {
        "beginner": "Focuses on understanding basic concepts and principles, with practical learning paths.",
        "intermediate": "Focuses on advanced concepts and practical project implementation, with paths to improve application skills.",
        "advanced": "Focuses on latest research trends and advanced technology implementation, with innovative approaches and expertise enhancement."
    }
    
    time_plans = {
        "short-term": "Proposes intensive learning plans centered on core skills and knowledge that can be acquired within 1-3 months.",
        "medium-term": "Proposes step-by-step learning plans to systematically build capabilities over 3-6 months.",
        "long-term": "Proposes comprehensive learning plans to develop expertise from a long-term perspective of 6 months to 1 year."
    }
    
    level_approach = level_approaches.get(learning_level, level_approaches["beginner"])
    time_plan = time_plans.get(time_horizon, time_plans["short-term"])
    
    output_template = f"""
    # {topic if topic else 'AI Trends'} Learning Guide
    
    ## 1. Trend Analysis
    - Key trends
    - Technological changes
    - Industry impact
    
    ## 2. Core Knowledge Areas
    - Basic concepts
    - Core technologies
    - Key algorithms/methodologies
    
    ## 3. Learning Roadmap
    - Step-by-step learning plan
    - Recommended resources
    - Practical projects
    
    ## 4. Career and Application Opportunities
    - Related roles/positions
    - Industry use cases
    - Future outlook
    """
    
    # Generate final prompt
    prompt = (
        f"You are an AI learning guide expert who analyzes the latest AI trends based on SPRI monthly AI reports "
        f"and provides customized learning directions.\n\n"
        
        f"## Learner Profile\n"
        f"- Level: {learning_level} ({level_approach})\n"
        f"- Learning Plan: {time_horizon} ({time_plan})\n\n"
        
        f"## Analysis Target\n"
        f"Please analyze based on the March issue of the SPRI monthly AI report. "
        f"{'Please focus on ' + topic + '-related content in your analysis.' if topic else 'Please analyze overall AI trends.'}\n\n"
        
        f"## Information to Provide\n"
        f"1. Summary of latest AI trends and their importance\n"
        f"2. Core knowledge and technical elements in the field\n"
        f"3. Step-by-step learning plan and recommended resources\n"
        f"4. Practical application suggestions and career recommendations\n\n"
        
        f"Please structure your analysis results as follows:\n\n{output_template}\n\n"
        
        f"Search the report to provide practical and specific information. Suggest learning directions "
        f"that align with the latest trends, and create a practical guide that learners can easily follow."
    )
    
    return prompt

@mcp.resource("help: hantaek@brain-crew.com")
def get_help() -> str:
    """Provides help for Dify knowledge search in Claude Desktop."""

    return """
            # Dify External Knowledge Search MCP Tool Usage Guide

            Enables Claude to search for information in documents using the Dify External Knowledge API.

            ## Available Tools

            1. search_knowledge - Search for information in the knowledge base
            - `query`: Search query
            - `top_k`: Maximum number of results to return (default: 5)
            - `score_threshold`: Minimum relevance score (default: 0.5)
            - `search_method`: Search method (semantic, keyword, hybrid) (default: hybrid)

            2. prompt(ai_trend_learning_guide)
            - `topic`: AI topic of interest (optional - e.g., "generative AI", "computer vision", "NLP")
            - `learning_level`: Learner level ("beginner", "intermediate", "advanced")
            - `time_horizon`: Learning plan duration ("short-term", "medium-term", "long-term")

            """

if __name__ == "__main__":
    mcp.run()