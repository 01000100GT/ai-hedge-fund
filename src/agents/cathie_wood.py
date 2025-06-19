"""
凯茜·伍德投资策略分析代理
该模块实现了基于凯茜·伍德投资理念的股票分析系统。主要关注:
1. 优先考虑具有突破性技术或商业模式的公司
2. 关注具有快速采用曲线和巨大总可寻址市场(TAM)的行业
3. 主要投资于AI、机器人、基因测序、金融科技和区块链
4. 愿意忍受短期波动以获取长期收益
"""

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm


class CathieWoodSignal(BaseModel):
    """
    凯茜·伍德风格的输出信号容器
    包含:
    - signal: 看涨/看跌/中性信号
    - confidence: 置信度(0-100)
    - reasoning: 推理说明
    """
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def cathie_wood_agent(state: AgentState):
    """
    使用凯茜·伍德的投资原则和LLM推理分析股票。
    1. 优先考虑具有突破性技术或商业模式的公司
    2. 关注具有快速采用曲线和巨大TAM的行业
    3. 主要投资于AI、机器人、基因测序、金融科技和区块链
    4. 愿意忍受短期波动以获取长期收益
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    cw_analysis = {}

    for ticker in tickers:
        progress.update_status("cathie_wood_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5)

        progress.update_status("cathie_wood_agent", ticker, "Gathering financial line items")
        # Request multiple periods of data (annual or TTM) for a more robust view.
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "gross_margin",
                "operating_margin",
                "debt_to_equity",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "outstanding_shares",
                "research_and_development",
                "capital_expenditure",
                "operating_expense",
            ],
            end_date,
            period="annual",
            limit=5,
        )

        progress.update_status("cathie_wood_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)

        progress.update_status("cathie_wood_agent", ticker, "Analyzing disruptive potential")
        disruptive_analysis = analyze_disruptive_potential(metrics, financial_line_items)

        progress.update_status("cathie_wood_agent", ticker, "Analyzing innovation-driven growth")
        innovation_analysis = analyze_innovation_growth(metrics, financial_line_items)

        progress.update_status("cathie_wood_agent", ticker, "Calculating valuation & high-growth scenario")
        valuation_analysis = analyze_cathie_wood_valuation(financial_line_items, market_cap)

        # Combine partial scores or signals
        total_score = disruptive_analysis["score"] + innovation_analysis["score"] + valuation_analysis["score"]
        max_possible_score = 15  # Adjust weighting as desired

        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {"signal": signal, "score": total_score, "max_score": max_possible_score, "disruptive_analysis": disruptive_analysis, "innovation_analysis": innovation_analysis, "valuation_analysis": valuation_analysis}

        progress.update_status("cathie_wood_agent", ticker, "Generating Cathie Wood analysis")
        cw_output = generate_cathie_wood_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
        )

        cw_analysis[ticker] = {"signal": cw_output.signal, "confidence": cw_output.confidence, "reasoning": cw_output.reasoning}

        progress.update_status("cathie_wood_agent", ticker, "Done", analysis=cw_output.reasoning)

    message = HumanMessage(content=json.dumps(cw_analysis), name="cathie_wood_agent")

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(cw_analysis, "Cathie Wood Agent")

    state["data"]["analyst_signals"]["cathie_wood_agent"] = cw_analysis

    progress.update_status("cathie_wood_agent", None, "Done")

    return {"messages": [message], "data": state["data"]}


def analyze_disruptive_potential(metrics: list, financial_line_items: list) -> dict:
    """
    分析公司是否具有颠覆性产品、技术或商业模式。
    评估颠覆性潜力的多个维度:
    1. 收入增长加速 - 表明市场采用
    2. 研发强度 - 显示创新投资
    3. 毛利率趋势 - 暗示定价能力和可扩展性
    4. 运营杠杆 - 展示商业模式效率
    5. 市场份额动态 - 表明竞争地位
    """
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze disruptive potential"}

    # 1. Revenue Growth Analysis - Check for accelerating growth
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    if len(revenues) >= 3:  # Need at least 3 periods to check acceleration
        growth_rates = []
        for i in range(len(revenues) - 1):
            if revenues[i] and revenues[i + 1]:
                growth_rate = (revenues[i] - revenues[i + 1]) / abs(revenues[i + 1]) if revenues[i + 1] != 0 else 0
                growth_rates.append(growth_rate)

        # Check if growth is accelerating (first growth rate higher than last, since they're in reverse order)
        if len(growth_rates) >= 2 and growth_rates[0] > growth_rates[-1]:
            score += 2
            details.append(f"Revenue growth is accelerating: {(growth_rates[0]*100):.1f}% vs {(growth_rates[-1]*100):.1f}%")

        # Check absolute growth rate (most recent growth rate is at index 0)
        latest_growth = growth_rates[0] if growth_rates else 0
        if latest_growth > 1.0:
            score += 3
            details.append(f"Exceptional revenue growth: {(latest_growth*100):.1f}%")
        elif latest_growth > 0.5:
            score += 2
            details.append(f"Strong revenue growth: {(latest_growth*100):.1f}%")
        elif latest_growth > 0.2:
            score += 1
            details.append(f"Moderate revenue growth: {(latest_growth*100):.1f}%")
    else:
        details.append("Insufficient revenue data for growth analysis")

    # 2. Gross Margin Analysis - Check for expanding margins
    gross_margins = [item.gross_margin for item in financial_line_items if hasattr(item, "gross_margin") and item.gross_margin is not None]
    if len(gross_margins) >= 2:
        margin_trend = gross_margins[0] - gross_margins[-1]
        if margin_trend > 0.05:  # 5% improvement
            score += 2
            details.append(f"Expanding gross margins: +{(margin_trend*100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append(f"Slightly improving gross margins: +{(margin_trend*100):.1f}%")

        # Check absolute margin level (most recent margin is at index 0)
        if gross_margins[0] > 0.50:  # High margin business
            score += 2
            details.append(f"High gross margin: {(gross_margins[0]*100):.1f}%")
    else:
        details.append("Insufficient gross margin data")

    # 3. Operating Leverage Analysis
    revenues = [item.revenue for item in financial_line_items if item.revenue]
    operating_expenses = [item.operating_expense for item in financial_line_items if hasattr(item, "operating_expense") and item.operating_expense]

    if len(revenues) >= 2 and len(operating_expenses) >= 2:
        rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
        opex_growth = (operating_expenses[0] - operating_expenses[-1]) / abs(operating_expenses[-1])

        if rev_growth > opex_growth:
            score += 2
            details.append("Positive operating leverage: Revenue growing faster than expenses")
    else:
        details.append("Insufficient data for operating leverage analysis")

    # 4. R&D Investment Analysis
    rd_expenses = [item.research_and_development for item in financial_line_items if hasattr(item, "research_and_development") and item.research_and_development is not None]
    if rd_expenses and revenues:
        rd_intensity = rd_expenses[0] / revenues[0]
        if rd_intensity > 0.15:  # High R&D intensity
            score += 3
            details.append(f"High R&D investment: {(rd_intensity*100):.1f}% of revenue")
        elif rd_intensity > 0.08:
            score += 2
            details.append(f"Moderate R&D investment: {(rd_intensity*100):.1f}% of revenue")
        elif rd_intensity > 0.05:
            score += 1
            details.append(f"Some R&D investment: {(rd_intensity*100):.1f}% of revenue")
    else:
        details.append("No R&D data available")

    # Normalize score to be out of 5
    max_possible_score = 12  # Sum of all possible points
    normalized_score = (score / max_possible_score) * 5

    return {"score": normalized_score, "details": "; ".join(details), "raw_score": score, "max_score": max_possible_score}


def analyze_innovation_growth(metrics: list, financial_line_items: list) -> dict:
    """
    评估公司对创新的承诺和指数增长潜力。
    分析多个维度:
    1. 研发投资趋势 - 衡量对创新的承诺
    2. 自由现金流生成 - 表明资助创新的能力
    3. 运营效率 - 显示创新的可扩展性
    4. 资本配置 - 揭示以创新为导向的管理层
    5. 增长再投资 - 展示对未来增长的承诺
    """
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze innovation-driven growth"}

    # 1. R&D Investment Trends
    rd_expenses = [item.research_and_development for item in financial_line_items if hasattr(item, "research_and_development") and item.research_and_development]
    revenues = [item.revenue for item in financial_line_items if item.revenue]

    if rd_expenses and revenues and len(rd_expenses) >= 2:
        rd_growth = (rd_expenses[0] - rd_expenses[-1]) / abs(rd_expenses[-1]) if rd_expenses[-1] != 0 else 0
        if rd_growth > 0.5:  # 50% growth in R&D
            score += 3
            details.append(f"Strong R&D investment growth: +{(rd_growth*100):.1f}%")
        elif rd_growth > 0.2:
            score += 2
            details.append(f"Moderate R&D investment growth: +{(rd_growth*100):.1f}%")

        # Check R&D intensity trend (corrected for reverse chronological order)
        rd_intensity_start = rd_expenses[-1] / revenues[-1]
        rd_intensity_end = rd_expenses[0] / revenues[0]
        if rd_intensity_end > rd_intensity_start:
            score += 2
            details.append(f"Increasing R&D intensity: {(rd_intensity_end*100):.1f}% vs {(rd_intensity_start*100):.1f}%")
    else:
        details.append("Insufficient R&D data for trend analysis")

    # 2. Free Cash Flow Analysis
    fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow]
    if fcf_vals and len(fcf_vals) >= 2:
        fcf_growth = (fcf_vals[0] - fcf_vals[-1]) / abs(fcf_vals[-1])
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)

        if fcf_growth > 0.3 and positive_fcf_count == len(fcf_vals):
            score += 3
            details.append("Strong and consistent FCF growth, excellent innovation funding capacity")
        elif positive_fcf_count >= len(fcf_vals) * 0.75:
            score += 2
            details.append("Consistent positive FCF, good innovation funding capacity")
        elif positive_fcf_count > len(fcf_vals) * 0.5:
            score += 1
            details.append("Moderately consistent FCF, adequate innovation funding capacity")
    else:
        details.append("Insufficient FCF data for analysis")

    # 3. Operating Efficiency Analysis
    op_margin_vals = [item.operating_margin for item in financial_line_items if item.operating_margin]
    if op_margin_vals and len(op_margin_vals) >= 2:
        margin_trend = op_margin_vals[0] - op_margin_vals[-1]

        if op_margin_vals[0] > 0.15 and margin_trend > 0:
            score += 3
            details.append(f"Strong and improving operating margin: {(op_margin_vals[0]*100):.1f}%")
        elif op_margin_vals[0] > 0.10:
            score += 2
            details.append(f"Healthy operating margin: {(op_margin_vals[0]*100):.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append("Improving operating efficiency")
    else:
        details.append("Insufficient operating margin data")

    # 4. Capital Allocation Analysis
    capex = [item.capital_expenditure for item in financial_line_items if hasattr(item, "capital_expenditure") and item.capital_expenditure]
    if capex and revenues and len(capex) >= 2:
        capex_intensity = abs(capex[0]) / revenues[0]
        capex_growth = (abs(capex[0]) - abs(capex[-1])) / abs(capex[-1]) if capex[-1] != 0 else 0

        if capex_intensity > 0.10 and capex_growth > 0.2:
            score += 2
            details.append("Strong investment in growth infrastructure")
        elif capex_intensity > 0.05:
            score += 1
            details.append("Moderate investment in growth infrastructure")
    else:
        details.append("Insufficient CAPEX data")

    # 5. Growth Reinvestment Analysis
    dividends = [item.dividends_and_other_cash_distributions for item in financial_line_items if hasattr(item, "dividends_and_other_cash_distributions") and item.dividends_and_other_cash_distributions]
    if dividends and fcf_vals:
        latest_payout_ratio = dividends[0] / fcf_vals[0] if fcf_vals[0] != 0 else 1
        if latest_payout_ratio < 0.2:  # Low dividend payout ratio suggests reinvestment focus
            score += 2
            details.append("Strong focus on reinvestment over dividends")
        elif latest_payout_ratio < 0.4:
            score += 1
            details.append("Moderate focus on reinvestment over dividends")
    else:
        details.append("Insufficient dividend data")

    # Normalize score to be out of 5
    max_possible_score = 15  # Sum of all possible points
    normalized_score = (score / max_possible_score) * 5

    return {"score": normalized_score, "details": "; ".join(details), "raw_score": score, "max_score": max_possible_score}


def analyze_cathie_wood_valuation(financial_line_items: list, market_cap: float) -> dict:
    """
    凯茜·伍德经常关注长期指数增长潜力。我们可以采用
    简化方法,寻找巨大的总可寻址市场(TAM)和公司
    捕获其中相当大部分的能力。
    """
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "Insufficient data for valuation"}

    latest = financial_line_items[0]
    fcf = latest.free_cash_flow if latest.free_cash_flow else 0

    if fcf <= 0:
        return {"score": 0, "details": f"No positive FCF for valuation; FCF = {fcf}", "intrinsic_value": None}

    # Instead of a standard DCF, let's assume a higher growth rate for an innovative company.
    # Example values:
    growth_rate = 0.20  # 20% annual growth
    discount_rate = 0.15
    terminal_multiple = 25
    projection_years = 5

    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv

    # Terminal Value
    terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) / ((1 + discount_rate) ** projection_years)
    intrinsic_value = present_value + terminal_value

    margin_of_safety = (intrinsic_value - market_cap) / market_cap

    score = 0
    if margin_of_safety > 0.5:
        score += 3
    elif margin_of_safety > 0.2:
        score += 1

    details = [f"Calculated intrinsic value: ~{intrinsic_value:,.2f}", f"Market cap: ~{market_cap:,.2f}", f"Margin of safety: {margin_of_safety:.2%}"]

    return {"score": score, "details": "; ".join(details), "intrinsic_value": intrinsic_value, "margin_of_safety": margin_of_safety}


def generate_cathie_wood_output(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
) -> CathieWoodSignal:
    """
    以凯茜·伍德的风格生成投资决策。
    """
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一个凯茜·伍德AI代理，使用她的原则做投资决策：

            1. 寻找利用颠覆性创新的公司。
            2. 强调指数增长潜力和巨大的总可寻址市场（TAM）。
            3. 专注于技术、医疗保健或其他面向未来的行业。
            4. 考虑多年时间跨度的潜在突破。
            5. 为追求高回报而接受更高的波动性。
            6. 评估管理层的愿景和投资研发的能力。

            规则：
            - 识别颠覆性或突破性技术。
            - 评估多年收入增长的强劲潜力。
            - 检查公司是否能在大市场中有效扩展。
            - 使用偏向增长的估值方法。
            - 提供数据驱动的建议（看涨、看跌或中性）。
            
            在提供推理时，要全面且具体：
            1. 识别公司正在利用的具体颠覆性技术/创新
            2. 突出表明指数潜力的增长指标（收入加速、扩大的TAM）
            3. 讨论5年以上时间跨度的长期愿景和变革潜力
            4. 解释公司如何可能颠覆传统行业或创造新市场
            5. 讨论可能推动未来增长的研发投资和创新管道
            6. 使用凯茜·伍德乐观、面向未来和充满信念的声音
            
            例如，如果看涨："该公司的AI驱动平台正在改变5000亿美元的医疗分析市场，平台采用率从40%加速到65%的同比增长证明了这一点。他们22%的收入用于研发投资正在创造技术护城河，使他们能够在这个不断扩大的市场中占据重要份额。当前估值没有反映我们预期的指数增长轨迹，因为..."
            例如，如果看跌："虽然在基因组学领域运营，但该公司缺乏真正的颠覆性技术，只是在逐步改进现有技术。研发支出仅占收入的8%，表明对突破性创新的投资不足。收入增长从45%放缓至20%，几乎没有证据表明我们在变革性公司中寻找的指数采用曲线..."
            """,
            ),
            (
                "human",
                """基于以下分析，创建凯茜·伍德风格的投资信号。

            {ticker}的分析数据：
            {analysis_data}

            请以此JSON格式返回交易信号：
            {{
              "signal": "bullish/bearish/neutral",
              "confidence": 浮点数 (0-100),
              "reasoning": "字符串"
            }}
            """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default_cathie_wood_signal():
        return CathieWoodSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=CathieWoodSignal,
        agent_name="cathie_wood_agent",
        state=state,
        default_factory=create_default_cathie_wood_signal,
    )


# source: https://ark-invest.com
