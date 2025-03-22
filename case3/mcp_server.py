import os
import requests

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP(
    name="Dify Workflow",
    version="0.0.1",
    description="Retrieve Dify Workflow execution results"
)

@mcp.tool()
async def dify_workflow(input: str) -> str:
    """
    Executes a Dify workflow and returns the results.
    Automates complex AI tasks and provides immediate results.
    Useful for text analysis, content generation, or processing user inputs through Dify workflows.
    
    Parameters:
        input: Input text to process
        
    """

    dify_base_url = os.getenv("DIFY_BASE_URL")
    dify_app_sk = os.getenv("DIFY_APP_SK")
    
    url = f"{dify_base_url}/workflows/run"
    headers = {
        "Authorization": f"Bearer {dify_app_sk}",
        "Content-Type": "application/json"
    }
    data = {
        "inputs": {"input": input},
        "response_mode": "blocking",
        "user": "default_user",
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    
    outputs = {}

    if "outputs" in result.get("data", {}):
        outputs = result["data"]["outputs"]
    
    return next(iter(outputs.values()), "No output received from Dify workflow.")

if __name__ == "__main__":
    mcp.run()