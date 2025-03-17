import argparse
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

def run_server(examples=None):
    """
    MCP 서버 - Claude RAG 통합 예제

    예제:
    1. 기본 RAG 구현
    2. Dify 외부지식 API
    3. Dify 워크플로우 연동

    """
    
    load_dotenv()
    
    mcp = FastMCP("Claude MCP Practice")
    
    if examples is None or "1" in examples:
        try:
            from example1.mcp import register_tools as register_example1
            register_example1(mcp)
            print("예제 1 도구 등록 완료")
        except Exception as e:
            print(f"예제 1 도구 등록 실패: {e}")
    
    if examples is None or "2" in examples:
        try:
            from example2.mcp import register_tools as register_example2
            register_example2(mcp)
            print("예제 2 도구 등록 완료")
        except Exception as e:
            print(f"예제 2 도구 등록 실패: {e}")
    
    if examples is None or "3" in examples:
        try:
            from example3.mcp import register_tools as register_example3
            register_example3(mcp)
            print("예제 3 도구 등록 완료")
        except Exception as e:
            print(f"예제 3 도구 등록 실패: {e}")
    
    examples_str = ", ".join(examples) if examples else "모두"
    print(f"MCP 서버 시작 (예제: {examples_str})")
    mcp.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claude RAG 예제 MCP 서버")
    parser.add_argument("--examples", nargs="+", choices=["1", "2", "3"],
                      help="실행할 예제 번호 (미지정시 모든 예제)")
    
    args = parser.parse_args()
    
    run_server(args.examples)