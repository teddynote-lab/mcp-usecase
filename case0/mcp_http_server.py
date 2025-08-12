from fastmcp import FastMCP

mcp = FastMCP("Demo 🚀")


@mcp.tool
def get_stock_code(stock_name: str) -> str:
    """Get stock code from stock name"""
    return "ABC123"


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8000, path="/mcp")
