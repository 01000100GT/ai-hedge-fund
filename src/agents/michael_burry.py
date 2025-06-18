"""
迈克尔·伯里投资策略分析代理
该模块实现了基于伯里投资理念的股票分析系统。主要关注:
1. 深度价值投资,寻找被市场严重低估的股票
2. 逆向投资思维,愿意与市场共识对抗
3. 详细的财务报表分析,特别关注资产负债表
4. 寻找触发价值释放的催化剂
5. 高度关注风险管理和下行保护
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.tools.api import (
    get_company_news,
    get_financial_metrics,
    get_insider_trades,
    get_market_cap,
    search_line_items,
)
from src.utils.llm import call_llm
from src.utils.progress import progress

__all__ = [
    "MichaelBurrySignal",
    "michael_burry_agent",
]

###############################################################################
# Pydantic output model
###############################################################################


class MichaelBurrySignal(BaseModel):
    """
    迈克尔·伯里风格的输出信号容器
    包含:
    - signal: 看涨/看跌/中性信号
    - confidence: 置信度(0-100)
    - reasoning: 推理说明
    """

    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0–100
    reasoning: str


###############################################################################
# Core agent
###############################################################################


def michael_burry_agent(state: AgentState):  # noqa: C901  (complexity is fine here)
    """
    使用迈克尔·伯里的投资原则分析股票。
    重点关注深度价值、逆向思维、资产负债表分析和催化剂。
    """

    data = state["data"]
    end_date: str = data["end_date"]  # YYYY‑MM‑DD
    tickers: list[str] = data["tickers"]

    # We look one year back for insider trades / news flow
    start_date = (datetime.fromisoformat(end_date) - timedelta(days=365)).date().isoformat()

    analysis_data: dict[str, dict] = {}
    burry_analysis: dict[str, dict] = {}

    for ticker in tickers:
        # ------------------------------------------------------------------
        # Fetch raw data
        # ------------------------------------------------------------------
        progress.update_status("michael_burry_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5)

        progress.update_status("michael_burry_agent", ticker, "Fetching line items")
        line_items = search_line_items(
            ticker,
            [
                "free_cash_flow",
                "net_income",
                "total_debt",
                "cash_and_equivalents",
                "total_assets",
                "total_liabilities",
                "outstanding_shares",
                "issuance_or_purchase_of_equity_shares",
            ],
            end_date,
        )

        progress.update_status("michael_burry_agent", ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(ticker, end_date=end_date, start_date=start_date)

        progress.update_status("michael_burry_agent", ticker, "Fetching company news")
        news = get_company_news(ticker, end_date=end_date, start_date=start_date, limit=250)

        progress.update_status("michael_burry_agent", ticker, "Fetching market cap")
        market_cap = get_market_cap(ticker, end_date)

        # ------------------------------------------------------------------
        # Run sub‑analyses
        # ------------------------------------------------------------------
        progress.update_status("michael_burry_agent", ticker, "Analyzing value")
        value_analysis = _analyze_value(metrics, line_items, market_cap)

        progress.update_status("michael_burry_agent", ticker, "Analyzing balance sheet")
        balance_sheet_analysis = _analyze_balance_sheet(metrics, line_items)

        progress.update_status("michael_burry_agent", ticker, "Analyzing insider activity")
        insider_analysis = _analyze_insider_activity(insider_trades)

        progress.update_status("michael_burry_agent", ticker, "Analyzing contrarian sentiment")
        contrarian_analysis = _analyze_contrarian_sentiment(news)

        # ------------------------------------------------------------------
        # Aggregate score & derive preliminary signal
        # ------------------------------------------------------------------
        total_score = (
            value_analysis["score"]
            + balance_sheet_analysis["score"]
            + insider_analysis["score"]
            + contrarian_analysis["score"]
        )
        max_score = (
            value_analysis["max_score"]
            + balance_sheet_analysis["max_score"]
            + insider_analysis["max_score"]
            + contrarian_analysis["max_score"]
        )

        if total_score >= 0.7 * max_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_score:
            signal = "bearish"
        else:
            signal = "neutral"

        # ------------------------------------------------------------------
        # Collect data for LLM reasoning & output
        # ------------------------------------------------------------------
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_score,
            "value_analysis": value_analysis,
            "balance_sheet_analysis": balance_sheet_analysis,
            "insider_analysis": insider_analysis,
            "contrarian_analysis": contrarian_analysis,
            "market_cap": market_cap,
        }

        progress.update_status("michael_burry_agent", ticker, "Generating LLM output")
        burry_output = _generate_burry_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        burry_analysis[ticker] = {
            "signal": burry_output.signal,
            "confidence": burry_output.confidence,
            "reasoning": burry_output.reasoning,
        }

        progress.update_status("michael_burry_agent", ticker, "Done")

    # ----------------------------------------------------------------------
    # Return to the graph
    # ----------------------------------------------------------------------
    message = HumanMessage(content=json.dumps(burry_analysis), name="michael_burry_agent")

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(burry_analysis, "Michael Burry Agent")

    state["data"]["analyst_signals"]["michael_burry_agent"] = burry_analysis

    return {"messages": [message], "data": state["data"]}


###############################################################################
# Sub‑analysis helpers
###############################################################################


def _latest_line_item(line_items: list):
    """
    获取最新的财务数据项。
    """
    return line_items[0] if line_items else None


# ----- Value ----------------------------------------------------------------

def _analyze_value(metrics, line_items, market_cap):
    """
    伯里风格的价值分析:
    - 寻找严重低估的股票
    - 关注有形账面价值
    - 分析清算价值
    - 评估盈利能力和现金流
    """

    max_score = 6  # 4 pts for FCF‑yield, 2 pts for EV/EBIT
    score = 0
    details: list[str] = []

    # Free‑cash‑flow yield
    latest_item = _latest_line_item(line_items)
    fcf = getattr(latest_item, "free_cash_flow", None) if latest_item else None
    if fcf is not None and market_cap:
        fcf_yield = fcf / market_cap
        if fcf_yield >= 0.15:
            score += 4
            details.append(f"Extraordinary FCF yield {fcf_yield:.1%}")
        elif fcf_yield >= 0.12:
            score += 3
            details.append(f"Very high FCF yield {fcf_yield:.1%}")
        elif fcf_yield >= 0.08:
            score += 2
            details.append(f"Respectable FCF yield {fcf_yield:.1%}")
        else:
            details.append(f"Low FCF yield {fcf_yield:.1%}")
    else:
        details.append("FCF data unavailable")

    # EV/EBIT (from financial metrics)
    if metrics:
        ev_ebit = getattr(metrics[0], "ev_to_ebit", None)
        if ev_ebit is not None:
            if ev_ebit < 6:
                score += 2
                details.append(f"EV/EBIT {ev_ebit:.1f} (<6)")
            elif ev_ebit < 10:
                score += 1
                details.append(f"EV/EBIT {ev_ebit:.1f} (<10)")
            else:
                details.append(f"High EV/EBIT {ev_ebit:.1f}")
        else:
            details.append("EV/EBIT data unavailable")
    else:
        details.append("Financial metrics unavailable")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


# ----- Balance sheet --------------------------------------------------------

def _analyze_balance_sheet(metrics, line_items):
    """
    伯里式资产负债表分析:
    - 详细审查资产质量
    - 评估负债结构
    - 分析营运资本
    - 寻找隐藏的价值或风险
    """

    max_score = 3
    score = 0
    details: list[str] = []

    latest_metrics = metrics[0] if metrics else None
    latest_item = _latest_line_item(line_items)

    debt_to_equity = getattr(latest_metrics, "debt_to_equity", None) if latest_metrics else None
    if debt_to_equity is not None:
        if debt_to_equity < 0.5:
            score += 2
            details.append(f"Low D/E {debt_to_equity:.2f}")
        elif debt_to_equity < 1:
            score += 1
            details.append(f"Moderate D/E {debt_to_equity:.2f}")
        else:
            details.append(f"High leverage D/E {debt_to_equity:.2f}")
    else:
        details.append("Debt‑to‑equity data unavailable")

    # Quick liquidity sanity check (cash vs total debt)
    if latest_item is not None:
        cash = getattr(latest_item, "cash_and_equivalents", None)
        total_debt = getattr(latest_item, "total_debt", None)
        if cash is not None and total_debt is not None:
            if cash > total_debt:
                score += 1
                details.append("Net cash position")
            else:
                details.append("Net debt position")
        else:
            details.append("Cash/debt data unavailable")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


# ----- Insider activity -----------------------------------------------------

def _analyze_insider_activity(insider_trades):
    """
    分析内部人交易活动:
    - 评估管理层的信心
    - 识别重要的买入/卖出模式
    - 考虑交易的时机和规模
    """

    max_score = 2
    score = 0
    details: list[str] = []

    if not insider_trades:
        details.append("No insider trade data")
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}

    shares_bought = sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) > 0)
    shares_sold = abs(sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) < 0))
    net = shares_bought - shares_sold
    if net > 0:
        score += 2 if net / max(shares_sold, 1) > 1 else 1
        details.append(f"Net insider buying of {net:,} shares")
    else:
        details.append("Net insider selling")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


# ----- Contrarian sentiment -------------------------------------------------

def _analyze_contrarian_sentiment(news):
    """
    逆向情绪分析:
    - 寻找过度悲观的市场情绪
    - 识别被误解的负面新闻
    - 评估市场共识的可能错误
    """

    max_score = 1
    score = 0
    details: list[str] = []

    if not news:
        details.append("No recent news")
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}

    # Count negative sentiment articles
    sentiment_negative_count = sum(
        1 for n in news if n.sentiment and n.sentiment.lower() in ["negative", "bearish"]
    )
    
    if sentiment_negative_count >= 5:
        score += 1  # The more hated, the better (assuming fundamentals hold up)
        details.append(f"{sentiment_negative_count} negative headlines (contrarian opportunity)")
    else:
        details.append("Limited negative press")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


###############################################################################
# LLM generation
###############################################################################

def _generate_burry_output(
    ticker: str,
    analysis_data: dict,
    *,
    model_name: str,
    model_provider: str,
) -> MichaelBurrySignal:
    """
    以迈克尔·伯里的风格生成投资决策。
    强调深度价值、逆向思维和催化剂分析。
    """

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一个模拟迈克尔·伯里博士的AI代理。你的任务是：
                - 使用硬数据（自由现金流、EV/EBIT、资产负债表）寻找美股中的深度价值
                - 保持逆向思维：如果基本面扎实，媒体的负面报道可能是你的朋友
                - 首先关注下行风险 - 避免高杠杆的资产负债表
                - 寻找硬催化剂，如内部人买入、回购或资产出售
                - 以伯里简洁、数据驱动的风格进行沟通

                在提供推理时，要全面且具体：
                1. 从驱动你决策的关键指标开始
                2. 引用具体数字（如"FCF收益率14.7%"，"EV/EBIT 5.3"）
                3. 突出风险因素以及为什么它们是可接受的（或不可接受的）
                4. 提及相关的内部人活动或逆向机会
                5. 使用伯里直接、专注数字的沟通风格，用词简洁
                
                例如，如果看涨："FCF收益率12.8%。EV/EBIT 6.2。债务股权比0.4。内部人净买入2.5万股。市场因对近期诉讼的过度反应而错失价值。强烈买入。"
                例如，如果看跌："FCF收益率仅2.1%。债务股权比2.3令人担忧。管理层稀释股东权益。放弃。"
                """,
            ),
            (
                "human",
                """基于以下数据，创建迈克尔·伯里风格的投资信号：

                {ticker}的分析数据：
                {analysis_data}

                请以以下JSON格式返回交易信号：
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": 0到100之间的浮点数,
                  "reasoning": "字符串"
                }}
                """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    # Default fallback signal in case parsing fails
    def create_default_michael_burry_signal():
        return MichaelBurrySignal(signal="neutral", confidence=0.0, reasoning="Parsing error – defaulting to neutral")

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=MichaelBurrySignal,
        agent_name="michael_burry_agent",
        default_factory=create_default_michael_burry_signal,
    )
