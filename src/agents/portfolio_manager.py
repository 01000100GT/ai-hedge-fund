"""
投资组合管理代理

该模块负责:
1. 整合各个分析师的投资信号
2. 根据风险限制和当前持仓做出最终交易决策
3. 生成具体的交易订单
4. 管理投资组合的整体风险和收益
"""

import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from src.graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm


class PortfolioDecision(BaseModel):
    """
    投资组合交易决策模型
    
    属性:
        action: 交易动作 - 买入/卖出/做空/回补/持有
        quantity: 交易数量(股数)
        confidence: 决策信心水平(0-100)
        reasoning: 决策理由说明
    """
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="交易股数")
    confidence: float = Field(description="决策信心水平,介于0.0和100.0之间")
    reasoning: str = Field(description="决策理由说明")


class PortfolioManagerOutput(BaseModel):
    """
    投资组合管理输出模型
    
    属性:
        decisions: 股票代码到交易决策的映射字典
    """
    decisions: dict[str, PortfolioDecision] = Field(description="股票代码到交易决策的映射字典")


##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState):
    """
    投资组合管理代理主函数
    
    为多个股票生成最终交易决策和订单。主要职责:
    1. 获取投资组合状态和分析师信号
    2. 获取每只股票的仓位限制、当前价格和信号
    3. 根据各种因素生成交易决策
    4. 输出详细的决策理由
    """

    # Get the portfolio and analyst signals
    portfolio = state["data"]["portfolio"]
    analyst_signals = state["data"]["analyst_signals"]
    tickers = state["data"]["tickers"]

    # Get position limits, current prices, and signals for every ticker
    position_limits = {}
    current_prices = {}
    max_shares = {}
    signals_by_ticker = {}
    for ticker in tickers:
        progress.update_status("portfolio_manager", ticker, "Processing analyst signals")

        # Get position limits and current prices for the ticker
        risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0)
        current_prices[ticker] = risk_data.get("current_price", 0)

        # Calculate maximum shares allowed based on position limit and price
        if current_prices[ticker] > 0:
            max_shares[ticker] = int(position_limits[ticker] / current_prices[ticker])
        else:
            max_shares[ticker] = 0

        # Get signals for the ticker
        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            if agent != "risk_management_agent" and ticker in signals:
                ticker_signals[agent] = {"signal": signals[ticker]["signal"], "confidence": signals[ticker]["confidence"]}
        signals_by_ticker[ticker] = ticker_signals

    progress.update_status("portfolio_manager", None, "Generating trading decisions")

    # Generate the trading decision
    result = generate_trading_decision(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
        state=state,
    )

    # Create the portfolio management message
    message = HumanMessage(
        content=json.dumps({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}),
        name="portfolio_manager",
    )

    # Print the decision if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}, "Portfolio Manager")

    progress.update_status("portfolio_manager", None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


def generate_trading_decision(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    current_prices: dict[str, float],
    max_shares: dict[str, int],
    portfolio: dict[str, float],
    state: AgentState,
) -> PortfolioManagerOutput:
    """
    尝试从LLM获取交易决策,包含重试逻辑
    
    参数:
        tickers: 股票代码列表
        signals_by_ticker: 每只股票的信号字典
        current_prices: 当前价格字典
        max_shares: 最大可交易股数字典
        portfolio: 投资组合状态
        model_name: 模型名称
        model_provider: 模型提供商
        
    返回:
        包含每只股票交易决策的PortfolioManagerOutput对象
    """
    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一名投资组合经理，基于多个股票代码做出最终交易决策。

              交易规则：
              - 对于多头仓位：
                * 只有在有可用现金时才买入
                * 只有在当前持有该股票多头股份时才卖出
                * 卖出数量必须 ≤ 当前多头仓位股数
                * 买入数量必须 ≤ 该股票的最大股数限制
              
              - 对于空头仓位：
                * 只有在有可用保证金时才做空（仓位价值 × 保证金要求）
                * 只有在当前持有该股票空头股份时才回补
                * 回补数量必须 ≤ 当前空头仓位股数
                * 做空数量必须遵守保证金要求
              
              - max_shares值已预先计算以遵守仓位限制
              - 根据信号考虑多头和空头机会
              - 对多头和空头敞口保持适当的风险管理

              可用操作：
              - "buy": 开仓或增加多头仓位
              - "sell": 平仓或减少多头仓位
              - "short": 开仓或增加空头仓位
              - "cover": 平仓或减少空头仓位
              - "hold": 无操作

              输入：
              - signals_by_ticker: 股票代码 → 信号的字典
              - max_shares: 每个股票允许的最大股数
              - portfolio_cash: 投资组合中的当前现金
              - portfolio_positions: 当前仓位（多头和空头）
              - current_prices: 每个股票的当前价格
              - margin_requirement: 空头仓位的当前保证金要求（例如0.5表示50%）
              - total_margin_used: 当前使用的总保证金
              """,
            ),
            (
                "human",
                """基于团队的分析，为每个股票代码做出交易决策。

              按股票代码分类的信号：
              {signals_by_ticker}

              当前价格：
              {current_prices}

              购买允许的最大股数：
              {max_shares}

              投资组合现金：{portfolio_cash}
              当前仓位：{portfolio_positions}
              当前保证金要求：{margin_requirement}
              已使用总保证金：{total_margin_used}

              严格按照以下结构输出JSON：
              {{
                "decisions": {{
                  "TICKER1": {{
                    "action": "buy/sell/short/cover/hold",
                    "quantity": 整数,
                    "confidence": 0到100之间的浮点数,
                    "reasoning": "字符串"
                  }},
                  "TICKER2": {{
                    ...
                  }},
                  ...
                }}
              }}
              """,
            ),
        ]
    )

    # Generate the prompt
    prompt = template.invoke(
        {
            "signals_by_ticker": json.dumps(signals_by_ticker, indent=2),
            "current_prices": json.dumps(current_prices, indent=2),
            "max_shares": json.dumps(max_shares, indent=2),
            "portfolio_cash": f"{portfolio.get('cash', 0):.2f}",
            "portfolio_positions": json.dumps(portfolio.get("positions", {}), indent=2),
            "margin_requirement": f"{portfolio.get('margin_requirement', 0):.2f}",
            "total_margin_used": f"{portfolio.get('margin_used', 0):.2f}",
        }
    )

    # Create default factory for PortfolioManagerOutput
    def create_default_portfolio_output():
        return PortfolioManagerOutput(decisions={ticker: PortfolioDecision(action="hold", quantity=0, confidence=0.0, reasoning="Error in portfolio management, defaulting to hold") for ticker in tickers})

    return call_llm(
        prompt=prompt,
        pydantic_model=PortfolioManagerOutput,
        agent_name="portfolio_manager",
        state=state,
        default_factory=create_default_portfolio_output,
    )
