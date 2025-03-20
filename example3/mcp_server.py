from mcp.server.fastmcp import FastMCP
import os
import requests
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP(name="dify_tools", version="1.0.0")

def register_tools(mcp):
    """MCP 서버에 도구 등록"""

    @mcp.tool()
    async def dify_workflow(input: str) -> str:
        """
        주어진 입력으로 Dify 워크플로우를 실행합니다.
        
        Args:
            input: 처리할 입력 텍스트
            
        Returns:
            Dify 워크플로우에서 반환된 결과

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
        
        return next(iter(outputs.values()), "Dify 워크플로우에서 출력을 받지 못했습니다.")