import datetime
import os
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from typing import Dict, Any, Optional, List

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)

# Global cache instance
_cache = get_cache()

# --- 配置带重试的 HTTP 会话 ---
# 创建一个 Session 对象
_session = requests.Session()

# 定义重试策略
# total=5: 总共重试 5 次
# backoff_factor=1: 等待时间因子 (1s, 2s, 4s, 8s, 16s)
# status_forcelist=[429, 500, 502, 503, 504]: 在这些状态码上触发重试
# allowed_methods=False: 对所有请求方法都应用重试（或指定 frozenset(['GET', 'POST'])）
retries = Retry(total=5, 
                backoff_factor=1, 
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=False) # 或者使用 method_whitelist=frozenset(['GET', 'POST'])

# 创建一个 HTTPAdapter 并挂载重试策略
adapter = HTTPAdapter(max_retries=retries)

# 将适配器挂载到 http:// 和 https://
_session.mount('http://', adapter)
_session.mount('https://', adapter)

# 全局设置 API Key (如果存在)，Session 会在所有请求中自动使用它
if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
    _session.headers.update({"X-API-KEY": api_key})

# --- Financial Data API Functions (使用 Session) ---

def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from cache or API using a session with retries."""
    # Check cache first
    if cached_data := _cache.get_prices(ticker):
        # Filter cached data by date range and convert to Price objects
        filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
        if filtered_data:
            return filtered_data

    # If not in cache or no data in range, fetch from API using the session
    # 注意: Session 已包含 API Key (如果设置了)
    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
    try:
        # 修改: 使用 _session.get 替代 requests.get
        response = _session.get(url, timeout=30) # 添加 timeout
        response.raise_for_status() # 检查 HTTP 错误状态码 (4xx, 5xx)
    except requests.exceptions.RequestException as e:
        # 处理 requests 相关的异常 (包括重试失败后的最终错误)
        print(f"Error fetching price data for {ticker} after retries: {e}")
        raise Exception(f"Error fetching data: {ticker} - {e}") from e

    # Parse response with Pydantic model
    price_response = PriceResponse(**response.json())
    prices = price_response.prices

    if not prices:
        return []

    # Cache the results as dicts
    _cache.set_prices(ticker, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API using a session with retries."""
    # Check cache first
    if cached_data := _cache.get_financial_metrics(ticker):
        # Filter cached data by date and limit
        filtered_data = [FinancialMetrics(**metric) for metric in cached_data if metric["report_period"] <= end_date]
        filtered_data.sort(key=lambda x: x.report_period, reverse=True)
        if filtered_data:
            return filtered_data[:limit]

    # If not in cache or insufficient data, fetch from API using the session
    url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
    try:
        # 修改: 使用 _session.get 替代 requests.get
        response = _session.get(url, timeout=30) # 添加 timeout
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching financial metrics for {ticker} after retries: {e}")
        raise Exception(f"Error fetching data: {ticker} - {e}") from e
        
    # Parse response with Pydantic model
    metrics_response = FinancialMetricsResponse(**response.json())
    # Return the FinancialMetrics objects directly instead of converting to dict
    financial_metrics = metrics_response.financial_metrics

    if not financial_metrics:
        return []

    # Cache the results as dicts
    _cache.set_financial_metrics(ticker, [m.model_dump() for m in financial_metrics])
    return financial_metrics


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Fetch line items from API using a session with retries."""
    url = "https://api.financialdatasets.ai/financials/search/line-items"
    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    try:
        # 修改: 使用 _session.post 替代 requests.post
        response = _session.post(url, json=body, timeout=30) # 添加 timeout
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error searching line items for {ticker} after retries: {e}")
        raise Exception(f"Error fetching data: {ticker} - {e}") from e
        
    data = response.json()
    response_model = LineItemResponse(**data)
    search_results = response_model.search_results
    if not search_results:
        return []

    # Cache the results
    return search_results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API using a session with retries."""
    # Check cache first
    if cached_data := _cache.get_insider_trades(ticker):
        # Filter cached data by date range
        filtered_data = [InsiderTrade(**trade) for trade in cached_data if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date) and (trade.get("transaction_date") or trade["filing_date"]) <= end_date]
        filtered_data.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    all_trades = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"
        try:
            # 修改: 使用 _session.get 替代 requests.get
            response = _session.get(url, timeout=30) # 添加 timeout
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching insider trades for {ticker} (page ending {current_end_date}) after retries: {e}")
            # 决定是抛出异常还是返回已获取的部分数据
            # 这里选择抛出异常，因为分页可能未完成
            raise Exception(f"Error fetching data: {ticker} - {e}") from e

        data = response.json()
        response_model = InsiderTradeResponse(**data)
        insider_trades = response_model.insider_trades

        if not insider_trades:
            break

        all_trades.extend(insider_trades)

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(insider_trades) < limit:
            break

        # Update end_date to the oldest filing date from current batch for next iteration
        current_end_date = min(trade.filing_date for trade in insider_trades).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_trades:
        return []

    # Cache the results
    _cache.set_insider_trades(ticker, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news from cache or API using a session with retries."""
    # 添加: 在每次调用前强制等待 1 秒，进行简单的节流
    print(f"[API Throttling] Waiting 1 second before fetching news for {ticker}...")
    time.sleep(1)

    # Check cache first
    if cached_data := _cache.get_company_news(ticker):
        # Filter cached data by date range
        filtered_data = [CompanyNews(**news) for news in cached_data if (start_date is None or news["date"] >= start_date) and news["date"] <= end_date]
        filtered_data.sort(key=lambda x: x.date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    all_news = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"
        try:
            # 修改: 使用 _session.get 替代 requests.get
            response = _session.get(url, timeout=30) # 添加 timeout
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching company news for {ticker} (page ending {current_end_date}) after retries: {e}")
            # 决定是抛出异常还是返回已获取的部分数据
            raise Exception(f"Error fetching data: {ticker} - {e}") from e
            
        data = response.json()
        response_model = CompanyNewsResponse(**data)
        company_news = response_model.news

        if not company_news:
            break

        all_news.extend(company_news)

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(company_news) < limit:
            break

        # Update end_date to the oldest date from current batch for next iteration
        current_end_date = min(news.date for news in company_news).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_news:
        return []

    # Cache the results
    _cache.set_company_news(ticker, [news.model_dump() for news in all_news])
    return all_news


def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """Fetch market cap from the API using a session with retries."""
    # Check if end_date is today
    if end_date == datetime.datetime.now().strftime("%Y-%m-%d"):
        url = f"https://api.financialdatasets.ai/company/facts/?ticker={ticker}"
        try:
            # 修改: 使用 _session.get 替代 requests.get
            response = _session.get(url, timeout=30) # 添加 timeout
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching company facts for {ticker} after retries: {e}")
            # 对于这种单点数据获取，可以考虑返回 None 而不是抛出异常
            return None

        data = response.json()
        response_model = CompanyFactsResponse(**data)
        return response_model.company_facts.market_cap

    financial_metrics = get_financial_metrics(ticker, end_date)
    if not financial_metrics:
        return None

    market_cap = financial_metrics[0].market_cap

    if not market_cap:
        return None

    return market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
