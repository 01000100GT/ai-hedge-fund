"""
数据模型定义模块
该模块定义了系统中使用的所有数据模型类，
包括价格、财务指标、内部交易、公司新闻等数据结构。
"""

from pydantic import BaseModel


class Price(BaseModel):
    """
    价格数据模型
    定义了单个时间点的价格信息结构
    """
    open: float
    close: float
    high: float
    low: float
    volume: int
    time: str


class PriceResponse(BaseModel):
    """
    价格数据响应模型
    包含股票代码和对应的价格数据列表
    """
    ticker: str
    prices: list[Price]


class FinancialMetrics(BaseModel):
    """
    财务指标模型
    包含公司的各项财务指标数据
    """
    ticker: str
    report_period: str
    period: str
    currency: str
    market_cap: float | None
    enterprise_value: float | None
    price_to_earnings_ratio: float | None
    price_to_book_ratio: float | None
    price_to_sales_ratio: float | None
    enterprise_value_to_ebitda_ratio: float | None
    enterprise_value_to_revenue_ratio: float | None
    free_cash_flow_yield: float | None
    peg_ratio: float | None
    gross_margin: float | None
    operating_margin: float | None
    net_margin: float | None
    return_on_equity: float | None
    return_on_assets: float | None
    return_on_invested_capital: float | None
    asset_turnover: float | None
    inventory_turnover: float | None
    receivables_turnover: float | None
    days_sales_outstanding: float | None
    operating_cycle: float | None
    working_capital_turnover: float | None
    current_ratio: float | None
    quick_ratio: float | None
    cash_ratio: float | None
    operating_cash_flow_ratio: float | None
    debt_to_equity: float | None
    debt_to_assets: float | None
    interest_coverage: float | None
    revenue_growth: float | None
    earnings_growth: float | None
    book_value_growth: float | None
    earnings_per_share_growth: float | None
    free_cash_flow_growth: float | None
    operating_income_growth: float | None
    ebitda_growth: float | None
    payout_ratio: float | None
    earnings_per_share: float | None
    book_value_per_share: float | None
    free_cash_flow_per_share: float | None


class FinancialMetricsResponse(BaseModel):
    """
    财务指标响应模型
    包含财务指标数据列表
    """
    financial_metrics: list[FinancialMetrics]


class LineItem(BaseModel):
    """
    财务报表行项目模型
    定义了财务报表中单个行项目的数据结构
    """
    ticker: str
    report_period: str
    period: str
    currency: str

    # Allow additional fields dynamically
    model_config = {"extra": "allow"}


class LineItemResponse(BaseModel):
    """
    财务报表行项目响应模型
    包含搜索结果列表
    """
    search_results: list[LineItem]


class InsiderTrade(BaseModel):
    """
    内部交易数据模型
    定义了公司内部人员交易信息的数据结构
    """
    ticker: str
    issuer: str | None
    name: str | None
    title: str | None
    is_board_director: bool | None
    transaction_date: str | None
    transaction_shares: float | None
    transaction_price_per_share: float | None
    transaction_value: float | None
    shares_owned_before_transaction: float | None
    shares_owned_after_transaction: float | None
    security_title: str | None
    filing_date: str


class InsiderTradeResponse(BaseModel):
    """
    内部交易响应模型
    包含内部交易数据列表
    """
    insider_trades: list[InsiderTrade]


class CompanyNews(BaseModel):
    """
    公司新闻数据模型
    定义了单条公司新闻的数据结构
    """
    ticker: str
    title: str
    author: str
    source: str
    date: str
    url: str
    sentiment: str | None = None


class CompanyNewsResponse(BaseModel):
    """
    公司新闻响应模型
    包含新闻数据列表
    """
    news: list[CompanyNews]


class CompanyFacts(BaseModel):
    """
    公司基本信息模型
    包含公司的基本信息数据
    """
    ticker: str
    name: str
    cik: str | None = None
    industry: str | None = None
    sector: str | None = None
    category: str | None = None
    exchange: str | None = None
    is_active: bool | None = None
    listing_date: str | None = None
    location: str | None = None
    market_cap: float | None = None
    number_of_employees: int | None = None
    sec_filings_url: str | None = None
    sic_code: str | None = None
    sic_industry: str | None = None
    sic_sector: str | None = None
    website_url: str | None = None
    weighted_average_shares: int | None = None


class CompanyFactsResponse(BaseModel):
    """
    公司基本信息响应模型
    包含公司基本信息数据
    """
    company_facts: CompanyFacts


class Position(BaseModel):
    """
    持仓位置模型
    定义了单个持仓的数据结构
    """
    cash: float = 0.0
    shares: int = 0
    ticker: str


class Portfolio(BaseModel):
    """
    投资组合模型
    包含所有持仓位置和总现金
    """
    positions: dict[str, Position]  # ticker -> Position mapping
    total_cash: float = 0.0


class AnalystSignal(BaseModel):
    """
    分析师信号模型
    定义了分析师产生的交易信号数据结构
    """
    signal: str | None = None
    confidence: float | None = None
    reasoning: dict | str | None = None
    max_position_size: float | None = None  # For risk management signals


class TickerAnalysis(BaseModel):
    """
    股票分析模型
    包含某只股票的所有分析师信号
    """
    ticker: str
    analyst_signals: dict[str, AnalystSignal]  # agent_name -> signal mapping


class AgentStateData(BaseModel):
    """
    智能体状态数据模型
    定义了智能体的状态数据结构
    """
    tickers: list[str]
    portfolio: Portfolio
    start_date: str
    end_date: str
    ticker_analyses: dict[str, TickerAnalysis]  # ticker -> analysis mapping


class AgentStateMetadata(BaseModel):
    """
    智能体状态元数据模型
    定义了智能体状态的元数据结构
    """
    show_reasoning: bool = False
    model_config = {"extra": "allow"}
