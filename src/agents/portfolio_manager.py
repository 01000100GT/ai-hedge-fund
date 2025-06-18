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
        progress.update_status("portfolio_management_agent", ticker, "Processing analyst signals")

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

    progress.update_status("portfolio_management_agent", None, "Generating trading decisions")

    # Generate the trading decision
    result = generate_trading_decision(
        tickers=tickers,
        signals_by_ticker=signals_by_ticker,
        current_prices=current_prices,
        max_shares=max_shares,
        portfolio=portfolio,
        model_name=state["metadata"]["model_name"],
        model_provider=state["metadata"]["model_provider"],
    )

    # Create the portfolio management message
    message = HumanMessage(
        content=json.dumps({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}),
        name="portfolio_management",
    )

    # Print the decision if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}, "Portfolio Management Agent")

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
    model_name: str,
    model_provider: str,
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
                """You are a portfolio manager making final trading decisions based on multiple tickers.

              Trading Rules:
              - For long positions:
                * Only buy if you have available cash
                * Only sell if you currently hold long shares of that ticker
                * Sell quantity must be ≤ current long position shares
                * Buy quantity must be ≤ max_shares for that ticker
              
              - For short positions:
                * Only short if you have available margin (position value × margin requirement)
                * Only cover if you currently have short shares of that ticker
                * Cover quantity must be ≤ current short position shares
                * Short quantity must respect margin requirements
              
              - The max_shares values are pre-calculated to respect position limits
              - Consider both long and short opportunities based on signals
              - Maintain appropriate risk management with both long and short exposure

              Available Actions:
              - "buy": Open or add to long position
              - "sell": Close or reduce long position
              - "short": Open or add to short position
              - "cover": Close or reduce short position
              - "hold": No action

              Inputs:
              - signals_by_ticker: dictionary of ticker → signals
              - max_shares: maximum shares allowed per ticker
              - portfolio_cash: current cash in portfolio
              - portfolio_positions: current positions (both long and short)
              - current_prices: current prices for each ticker
              - margin_requirement: current margin requirement for short positions (e.g., 0.5 means 50%)
              - total_margin_used: total margin currently in use
              """,
            ),
            (
                "human",
                """Based on the team's analysis, make your trading decisions for each ticker.

              Here are the signals by ticker:
              {signals_by_ticker}

              Current Prices:
              {current_prices}

              Maximum Shares Allowed For Purchases:
              {max_shares}

              Portfolio Cash: {portfolio_cash}
              Current Positions: {portfolio_positions}
              Current Margin Requirement: {margin_requirement}
              Total Margin Used: {total_margin_used}

              Output strictly in JSON with the following structure:
              {{
                "decisions": {{
                  "TICKER1": {{
                    "action": "buy/sell/short/cover/hold",
                    "quantity": integer,
                    "confidence": float between 0 and 100,
                    "reasoning": "string"
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

    return call_llm(prompt=prompt, model_name=model_name, model_provider=model_provider, pydantic_model=PortfolioManagerOutput, agent_name="portfolio_management_agent", default_factory=create_default_portfolio_output)
