from fastmcp import FastMCP
from pykrx import stock
from datetime import datetime, timedelta
from typing import Optional, List, Dict

# 주식 시장 데이터 MCP 서버
mcp = FastMCP(name="Stock Market MCP Server", version="1.0.0")


# 현재가 조회
@mcp.tool
def get_stock_price(ticker: str) -> dict:
    """Get current stock price information for a given ticker

    Args:
        ticker: Stock ticker code (e.g., '005930' for Samsung Electronics)

    Returns:
        Current day OHLCV data with price changes
    """
    try:
        today = datetime.now().strftime("%Y%m%d")
        df = stock.get_market_ohlcv(today, today, ticker)

        if df.empty:
            # 오늘 데이터가 없으면 이전 영업일 데이터 확인
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
            df = stock.get_market_ohlcv(yesterday, yesterday, ticker)
            if df.empty:
                return {"error": f"No data available for ticker {ticker}"}
            today = yesterday

        result = df.iloc[0].to_dict()

        return {
            "종목코드": ticker,
            "날짜": today,
            "시가": int(result.get("시가", 0)),
            "고가": int(result.get("고가", 0)),
            "저가": int(result.get("저가", 0)),
            "종가": int(result.get("종가", 0)),
            "거래량": int(result.get("거래량", 0)),
            "거래대금": (
                int(result.get("거래대금", 0)) if "거래대금" in result else None
            ),
            "등락률": float(result.get("등락률", 0)) if "등락률" in result else None,
        }

    except Exception as e:
        return {"error": f"Failed to fetch stock data: {str(e)}"}


# 기간별 주가 데이터 조회
@mcp.tool
def get_stock_history(
    ticker: str, start_date: str, end_date: Optional[str] = None
) -> dict:
    """Get historical stock price data for a period

    Args:
        ticker: Stock ticker code
        start_date: Start date (YYYYMMDD)
        end_date: End date (YYYYMMDD), defaults to today if not provided

    Returns:
        Historical OHLCV data for the specified period
    """
    try:
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")

        df = stock.get_market_ohlcv(start_date, end_date, ticker)

        if df.empty:
            return {
                "error": f"No data available for ticker {ticker} in the specified period"
            }

        # 데이터프레임을 딕셔너리 리스트로 변환
        records = []
        for date, row in df.iterrows():
            records.append(
                {
                    "날짜": date.strftime("%Y-%m-%d"),
                    "시가": int(row["시가"]),
                    "고가": int(row["고가"]),
                    "저가": int(row["저가"]),
                    "종가": int(row["종가"]),
                    "거래량": int(row["거래량"]),
                }
            )

        return {
            "종목코드": ticker,
            "기간": f"{start_date} ~ {end_date}",
            "데이터수": len(records),
            "주가데이터": records,
        }

    except Exception as e:
        return {"error": f"Failed to fetch historical data: {str(e)}"}


# 시가총액 순위 조회
@mcp.tool
def get_market_cap_ranking(market: str = "KOSPI", top_n: int = 20) -> dict:
    """Get top stocks by market capitalization

    Args:
        market: Market type (KOSPI, KOSDAQ, KONEX)
        top_n: Number of top stocks to return

    Returns:
        Top stocks ranked by market cap with fundamental data
    """
    try:
        today = datetime.now().strftime("%Y%m%d")
        df = stock.get_market_cap(today, market=market)

        if df.empty:
            # 이전 영업일 데이터 확인
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
            df = stock.get_market_cap(yesterday, market=market)
            if df.empty:
                return {"error": "No market cap data available"}
            today = yesterday

        # 시가총액 기준 정렬 및 상위 N개 선택
        df = df.sort_values("시가총액", ascending=False).head(top_n)

        results = []
        for ticker, row in df.iterrows():
            name = stock.get_market_ticker_name(ticker)
            results.append(
                {
                    "순위": len(results) + 1,
                    "종목코드": ticker,
                    "종목명": name,
                    "시가총액": int(row["시가총액"]),
                    "상장주식수": int(row["상장주식수"]),
                    "종가": int(row["종가"]),
                }
            )

        return {
            "시장": market,
            "기준일": today,
            "상위종목수": len(results),
            "종목리스트": results,
        }

    except Exception as e:
        return {"error": f"Failed to fetch market cap ranking: {str(e)}"}


# 거래량 순위 조회
@mcp.tool
def get_volume_ranking(market: str = "KOSPI", top_n: int = 20) -> dict:
    """Get top stocks by trading volume

    Args:
        market: Market type (KOSPI, KOSDAQ)
        top_n: Number of top stocks to return

    Returns:
        Top stocks ranked by trading volume
    """
    try:
        today = datetime.now().strftime("%Y%m%d")
        df = stock.get_market_ohlcv(today, market=market)

        if df.empty:
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
            df = stock.get_market_ohlcv(yesterday, market=market)
            if df.empty:
                return {"error": "No trading data available"}
            today = yesterday

        # 거래량 기준 정렬 및 상위 N개 선택
        df = df.sort_values("거래량", ascending=False).head(top_n)

        results = []
        for ticker, row in df.iterrows():
            name = stock.get_market_ticker_name(ticker)
            results.append(
                {
                    "순위": len(results) + 1,
                    "종목코드": ticker,
                    "종목명": name,
                    "거래량": int(row["거래량"]),
                    "거래대금": int(row["거래대금"]),
                    "종가": int(row["종가"]),
                    "등락률": float(row["등락률"]),
                }
            )

        return {
            "시장": market,
            "기준일": today,
            "상위종목수": len(results),
            "종목리스트": results,
        }

    except Exception as e:
        return {"error": f"Failed to fetch volume ranking: {str(e)}"}


# 상승률 상위 종목
@mcp.tool
def get_top_gainers(market: str = "KOSPI", top_n: int = 20) -> dict:
    """Get top gaining stocks by price change rate

    Args:
        market: Market type (KOSPI, KOSDAQ)
        top_n: Number of top stocks to return

    Returns:
        Top gaining stocks for the day
    """
    try:
        today = datetime.now().strftime("%Y%m%d")
        df = stock.get_market_ohlcv(today, market=market)

        if df.empty:
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
            df = stock.get_market_ohlcv(yesterday, market=market)
            if df.empty:
                return {"error": "No trading data available"}
            today = yesterday

        # 등락률 기준 정렬 및 상승 상위 종목 선택
        df = df.sort_values("등락률", ascending=False).head(top_n)

        results = []
        for ticker, row in df.iterrows():
            name = stock.get_market_ticker_name(ticker)
            results.append(
                {
                    "순위": len(results) + 1,
                    "종목코드": ticker,
                    "종목명": name,
                    "현재가": int(row["종가"]),
                    "등락률": f"{float(row['등락률']):.2f}%",
                    "거래량": int(row["거래량"]),
                }
            )

        return {
            "시장": market,
            "기준일": today,
            "상승종목수": len(results),
            "종목리스트": results,
        }

    except Exception as e:
        return {"error": f"Failed to fetch top gainers: {str(e)}"}


# 하락률 상위 종목
@mcp.tool
def get_top_losers(market: str = "KOSPI", top_n: int = 20) -> dict:
    """Get top losing stocks by price change rate

    Args:
        market: Market type (KOSPI, KOSDAQ)
        top_n: Number of top stocks to return

    Returns:
        Top losing stocks for the day
    """
    try:
        today = datetime.now().strftime("%Y%m%d")
        df = stock.get_market_ohlcv(today, market=market)

        if df.empty:
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
            df = stock.get_market_ohlcv(yesterday, market=market)
            if df.empty:
                return {"error": "No trading data available"}
            today = yesterday

        # 등락률 기준 정렬 및 하락 상위 종목 선택
        df = df.sort_values("등락률", ascending=True).head(top_n)

        results = []
        for ticker, row in df.iterrows():
            name = stock.get_market_ticker_name(ticker)
            results.append(
                {
                    "순위": len(results) + 1,
                    "종목코드": ticker,
                    "종목명": name,
                    "현재가": int(row["종가"]),
                    "등락률": f"{float(row['등락률']):.2f}%",
                    "거래량": int(row["거래량"]),
                }
            )

        return {
            "시장": market,
            "기준일": today,
            "하락종목수": len(results),
            "종목리스트": results,
        }

    except Exception as e:
        return {"error": f"Failed to fetch top losers: {str(e)}"}


# 투자자별 매매 동향
@mcp.tool
def get_investor_trading(
    ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None
) -> dict:
    """Get investor trading trends (foreign/institutional/individual)

    Args:
        ticker: Stock ticker code
        start_date: Start date (YYYYMMDD), defaults to 30 days ago
        end_date: End date (YYYYMMDD), defaults to today

    Returns:
        Trading volume and value by investor type
    """
    try:
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

        df = stock.get_market_trading_value_by_investor(start_date, end_date, ticker)

        if df.empty:
            return {"error": f"No investor trading data available for {ticker}"}

        # 기간 내 총 거래대금 계산
        summary = {
            "기관": df["기관"].sum(),
            "외국인": (
                df["외인"].sum()
                if "외인" in df.columns
                else df["외국인"].sum() if "외국인" in df.columns else 0
            ),
            "개인": df["개인"].sum(),
            "기타법인": df.get("기타법인", 0).sum() if "기타법인" in df.columns else 0,
        }

        # 최근 10일 데이터 추출
        recent_data = []
        for date, row in df.tail(10).iterrows():
            recent_data.append(
                {
                    "날짜": date.strftime("%Y-%m-%d"),
                    "기관": int(row.get("기관", 0)),
                    "외국인": int(row.get("외인", row.get("외국인", 0))),
                    "개인": int(row.get("개인", 0)),
                }
            )

        return {
            "종목코드": ticker,
            "종목명": stock.get_market_ticker_name(ticker),
            "기간": f"{start_date} ~ {end_date}",
            "누적매매대금": {
                "기관": int(summary["기관"]),
                "외국인": int(summary["외국인"]),
                "개인": int(summary["개인"]),
                "기타법인": int(summary["기타법인"]),
            },
            "최근10일": recent_data,
        }

    except Exception as e:
        return {"error": f"Failed to fetch investor trading data: {str(e)}"}


# 재무 지표 조회
@mcp.tool
def get_stock_fundamental(ticker: str) -> dict:
    """Get fundamental indicators for a stock (PER, PBR, EPS, etc.)

    Args:
        ticker: Stock ticker code

    Returns:
        Fundamental indicators including PER, PBR, EPS, dividend yield
    """
    try:
        today = datetime.now().strftime("%Y%m%d")
        df = stock.get_market_fundamental(today, today, ticker)

        if df.empty:
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
            df = stock.get_market_fundamental(yesterday, yesterday, ticker)
            if df.empty:
                return {"error": f"No fundamental data available for {ticker}"}
            today = yesterday

        result = df.iloc[0].to_dict()

        return {
            "종목코드": ticker,
            "종목명": stock.get_market_ticker_name(ticker),
            "기준일": today,
            "재무지표": {
                "BPS": int(result.get("BPS", 0)),
                "PER": float(result.get("PER", 0)),
                "PBR": float(result.get("PBR", 0)),
                "EPS": int(result.get("EPS", 0)),
                "DIV": float(result.get("DIV", 0)),
                "DPS": int(result.get("DPS", 0)),
            },
            "설명": {
                "BPS": "주당순자산가치",
                "PER": "주가수익비율",
                "PBR": "주가순자산비율",
                "EPS": "주당순이익",
                "DIV": "배당수익률(%)",
                "DPS": "주당배당금",
            },
        }

    except Exception as e:
        return {"error": f"Failed to fetch fundamental data: {str(e)}"}


# 종목명 검색
@mcp.tool
def search_stock_by_name(keyword: str, market: Optional[str] = None) -> dict:
    """Search stock ticker by company name

    Args:
        keyword: Company name or part of it
        market: Market type (KOSPI, KOSDAQ), searches all if not specified

    Returns:
        List of matching stocks with ticker codes
    """
    try:
        markets_to_search = []
        if market:
            markets_to_search = [market]
        else:
            markets_to_search = ["KOSPI", "KOSDAQ"]

        all_results = []

        for mkt in markets_to_search:
            tickers = stock.get_market_ticker_list(market=mkt)

            for ticker in tickers:
                name = stock.get_market_ticker_name(ticker)
                if keyword.upper() in name.upper():
                    all_results.append(
                        {"종목코드": ticker, "종목명": name, "시장": mkt}
                    )

        if not all_results:
            return {"error": f"No stocks found matching '{keyword}'"}

        return {
            "검색어": keyword,
            "검색결과수": len(all_results),
            "종목리스트": all_results,
        }

    except Exception as e:
        return {"error": f"Failed to search stocks: {str(e)}"}


# 시장 전체 종목 리스트
@mcp.tool
def get_market_tickers(market: str = "KOSPI", include_etf: bool = False) -> dict:
    """Get all ticker codes in a market

    Args:
        market: Market type (KOSPI, KOSDAQ, KONEX)
        include_etf: Whether to include ETF tickers

    Returns:
        List of all tickers with names in the specified market
    """
    try:
        tickers = stock.get_market_ticker_list(market=market)

        results = []
        for ticker in tickers:
            name = stock.get_market_ticker_name(ticker)

            # ETF 포함하지 않는 경우 건너뛰기
            if not include_etf and "ETF" in name.upper():
                continue

            results.append({"종목코드": ticker, "종목명": name})

        return {
            "시장": market,
            "ETF포함": include_etf,
            "종목수": len(results),
            "종목리스트": results,
        }

    except Exception as e:
        return {"error": f"Failed to fetch market tickers: {str(e)}"}


if __name__ == "__main__":
    mcp.run()
