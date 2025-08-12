from fastmcp import FastMCP
from fastmcp.prompts.prompt import PromptMessage, TextContent


# 샘플 MCP 서버
mcp = FastMCP(name="Sample MCP Server", version="1.0.0")


# 리소스 - 애플리케이션 상태 정보를 제공하는 리소스
@mcp.resource(
    uri="data://app-status",  # 명시적 URI (필수)
    name="ApplicationStatus",  # 사용자 정의 이름
    mime_type="application/json",  # MIME 타입
)
def get_application_status() -> dict:
    """Get application status information."""
    return {
        "status": "ok",
        "uptime": 12345,
        "company": "TeddyNote",
    }


# 프롬프트 - 특정 주제에 대한 설명을 요청하는 프롬프트
@mcp.prompt
def ask_about_topic(topic: str) -> str:
    """Generates a user message asking for an explanation of a topic."""
    return f"Can you please explain the concept of '{topic}'?"


# 특정 메시지 타입을 반환하는 프롬프트
@mcp.prompt
def generate_code_request(language: str, task_description: str) -> PromptMessage:
    """Generates a user message requesting code generation."""
    content = f"Write a {language} function that performs the following task: {task_description}"
    return PromptMessage(role="user", content=TextContent(type="text", text=content))


@mcp.tool
def get_stock_code(stock_name: str) -> str:
    """Get stock code from stock name."""

    return "ABC123"


if __name__ == "__main__":
    mcp.run()
