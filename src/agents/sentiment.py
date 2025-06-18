"""
市场情绪分析代理

该模块负责:
1. 分析市场情绪并生成交易信号
2. 整合内部交易数据和公司新闻
3. 计算加权情绪得分
4. 生成详细的情绪分析报告
"""

from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
import pandas as pd
import numpy as np
import json

from src.tools.api import get_insider_trades, get_company_news


def sentiment_agent(state: AgentState):
    """
    市场情绪分析代理主函数
    
    分析市场情绪并为多个股票生成交易信号。主要职责:
    1. 获取内部交易数据
    2. 分析交易模式
    3. 获取并分析公司新闻
    4. 结合多个信号源生成整体情绪评估
    5. 计算加权信号和信心水平
    """
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")

    # Initialize sentiment analysis for each ticker
    sentiment_analysis = {}

    for ticker in tickers:
        progress.update_status("sentiment_agent", ticker, "Fetching insider trades")

        # Get the insider trades
        insider_trades = get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            limit=1000,
        )

        progress.update_status("sentiment_agent", ticker, "Analyzing trading patterns")

        # Get the signals from the insider trades
        transaction_shares = pd.Series([t.transaction_shares for t in insider_trades]).dropna()
        insider_signals = np.where(transaction_shares < 0, "bearish", "bullish").tolist()

        progress.update_status("sentiment_agent", ticker, "Fetching company news")

        # Get the company news
        company_news = get_company_news(ticker, end_date, limit=100)

        # Get the sentiment from the company news
        sentiment = pd.Series([n.sentiment for n in company_news]).dropna()
        news_signals = np.where(sentiment == "negative", "bearish", 
                              np.where(sentiment == "positive", "bullish", "neutral")).tolist()
        
        progress.update_status("sentiment_agent", ticker, "Combining signals")
        # Combine signals from both sources with weights
        insider_weight = 0.3  # 内部交易权重
        news_weight = 0.7    # 新闻情绪权重
        
        # Calculate weighted signal counts
        bullish_signals = (
            insider_signals.count("bullish") * insider_weight +
            news_signals.count("bullish") * news_weight
        )
        bearish_signals = (
            insider_signals.count("bearish") * insider_weight +
            news_signals.count("bearish") * news_weight
        )

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level based on the weighted proportion
        total_weighted_signals = len(insider_signals) * insider_weight + len(news_signals) * news_weight
        confidence = 0  # Default confidence when there are no signals
        if total_weighted_signals > 0:
            confidence = round(max(bullish_signals, bearish_signals) / total_weighted_signals, 2) * 100
        reasoning = f"加权看涨信号: {bullish_signals:.1f}, 加权看跌信号: {bearish_signals:.1f}"

        sentiment_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status("sentiment_agent", ticker, "Done")

    # Create the sentiment message
    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name="sentiment_agent",
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "Sentiment Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["sentiment_agent"] = sentiment_analysis

    return {
        "messages": [message],
        "data": data,
    }
