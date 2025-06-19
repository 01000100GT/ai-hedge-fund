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
    open: float  # 开盘价：交易日开始时的股票价格
    close: float  # 收盘价：交易日结束时的股票价格
    high: float  # 最高价：交易日内股票达到的最高价格
    low: float  # 最低价：交易日内股票达到的最低价格
    volume: int  # 成交量：当日股票交易的总股数
    time: str  # 时间戳：价格数据对应的时间点（格式：YYYY-MM-DD）


class PriceResponse(BaseModel):
    """
    价格数据响应模型
    包含股票代码和对应的价格数据列表
    """
    ticker: str  # 股票代码：股票的唯一标识符（如：AAPL、TSLA）
    prices: list[Price]  # 价格数据列表：包含多个时间点的价格信息


class FinancialMetrics(BaseModel):
    """
    财务指标模型
    包含公司的各项财务指标数据
    """
    ticker: str  # 股票代码：股票的唯一标识符
    report_period: str  # 报告期间：财务报告的时间范围
    period: str  # 周期类型：如季度(Q)、年度(A)等
    currency: str  # 货币单位：财务数据使用的货币（如USD、CNY）
    
    # 估值指标
    market_cap: float | None  # 市值：公司股票总市值（股价×总股本）
    enterprise_value: float | None  # 企业价值：市值+净债务的总价值
    price_to_earnings_ratio: float | None  # 市盈率(P/E)：股价与每股收益的比率
    price_to_book_ratio: float | None  # 市净率(P/B)：股价与每股净资产的比率
    price_to_sales_ratio: float | None  # 市销率(P/S)：股价与每股销售收入的比率
    enterprise_value_to_ebitda_ratio: float | None  # EV/EBITDA：企业价值与息税折旧前利润的比率
    enterprise_value_to_revenue_ratio: float | None  # EV/Revenue：企业价值与收入的比率
    free_cash_flow_yield: float | None  # 自由现金流收益率：自由现金流与市值的比率
    peg_ratio: float | None  # PEG比率：市盈率与盈利增长率的比率
    
    # 盈利能力指标
    gross_margin: float | None  # 毛利率：毛利润占收入的百分比
    operating_margin: float | None  # 营业利润率：营业利润占收入的百分比
    net_margin: float | None  # 净利润率：净利润占收入的百分比
    return_on_equity: float | None  # 净资产收益率(ROE)：净利润与股东权益的比率
    return_on_assets: float | None  # 资产收益率(ROA)：净利润与总资产的比率
    return_on_invested_capital: float | None  # 投入资本回报率(ROIC)：税后营业利润与投入资本的比率
    
    # 运营效率指标
    asset_turnover: float | None  # 资产周转率：收入与平均总资产的比率
    inventory_turnover: float | None  # 存货周转率：销售成本与平均存货的比率
    receivables_turnover: float | None  # 应收账款周转率：收入与平均应收账款的比率
    days_sales_outstanding: float | None  # 应收账款天数：应收账款的平均收回天数
    operating_cycle: float | None  # 营业周期：存货周转天数+应收账款周转天数
    working_capital_turnover: float | None  # 营运资本周转率：收入与营运资本的比率
    
    # 流动性指标
    current_ratio: float | None  # 流动比率：流动资产与流动负债的比率
    quick_ratio: float | None  # 速动比率：速动资产与流动负债的比率
    cash_ratio: float | None  # 现金比率：现金及现金等价物与流动负债的比率
    operating_cash_flow_ratio: float | None  # 经营现金流比率：经营现金流与流动负债的比率
    
    # 杠杆指标
    debt_to_equity: float | None  # 负债权益比：总负债与股东权益的比率
    debt_to_assets: float | None  # 负债资产比：总负债与总资产的比率
    interest_coverage: float | None  # 利息保障倍数：息税前利润与利息费用的比率
    
    # 增长指标
    revenue_growth: float | None  # 收入增长率：当期收入相对上期的增长百分比
    earnings_growth: float | None  # 盈利增长率：当期净利润相对上期的增长百分比
    book_value_growth: float | None  # 账面价值增长率：每股净资产的增长百分比
    earnings_per_share_growth: float | None  # 每股收益增长率：EPS的增长百分比
    free_cash_flow_growth: float | None  # 自由现金流增长率：自由现金流的增长百分比
    operating_income_growth: float | None  # 营业收入增长率：营业收入的增长百分比
    ebitda_growth: float | None  # EBITDA增长率：息税折旧前利润的增长百分比
    
    # 每股指标
    payout_ratio: float | None  # 股息支付率：股息与净利润的比率
    earnings_per_share: float | None  # 每股收益(EPS)：净利润除以流通股数
    book_value_per_share: float | None  # 每股净资产：股东权益除以流通股数
    free_cash_flow_per_share: float | None  # 每股自由现金流：自由现金流除以流通股数


class FinancialMetricsResponse(BaseModel):
    """
    财务指标响应模型
    包含财务指标数据列表
    """
    financial_metrics: list[FinancialMetrics]  # 财务指标列表：包含多个财务指标数据


class LineItem(BaseModel):
    """
    财务报表行项目模型
    定义了财务报表中单个行项目的数据结构
    """
    ticker: str  # 股票代码：股票的唯一标识符
    report_period: str  # 报告期间：财务报告的时间范围
    period: str  # 周期类型：如季度(Q)、年度(A)等
    currency: str  # 货币单位：财务数据使用的货币

    # Allow additional fields dynamically
    model_config = {"extra": "allow"}  # 允许额外字段：支持动态添加其他财务报表项目


class LineItemResponse(BaseModel):
    """
    财务报表行项目响应模型
    包含搜索结果列表
    """
    search_results: list[LineItem]  # 搜索结果：包含匹配的财务报表行项目列表


class InsiderTrade(BaseModel):
    """
    内部交易数据模型
    定义了公司内部人员交易信息的数据结构
    """
    ticker: str  # 股票代码：交易涉及的股票标识符
    issuer: str | None  # 发行人：股票发行公司名称
    name: str | None  # 交易人姓名：进行内部交易的人员姓名
    title: str | None  # 职位头衔：交易人在公司的职务
    is_board_director: bool | None  # 是否为董事：交易人是否为公司董事会成员
    transaction_date: str | None  # 交易日期：内部交易发生的日期
    transaction_shares: float | None  # 交易股数：本次交易涉及的股票数量
    transaction_price_per_share: float | None  # 每股交易价格：交易时的股票单价
    transaction_value: float | None  # 交易总价值：交易股数×每股价格
    shares_owned_before_transaction: float | None  # 交易前持股数：交易前该人员持有的股票数量
    shares_owned_after_transaction: float | None  # 交易后持股数：交易后该人员持有的股票数量
    security_title: str | None  # 证券类型：交易证券的类型（如普通股、期权等）
    filing_date: str  # 申报日期：向监管机构提交交易报告的日期


class InsiderTradeResponse(BaseModel):
    """
    内部交易响应模型
    包含内部交易数据列表
    """
    insider_trades: list[InsiderTrade]  # 内部交易列表：包含多条内部交易记录


class CompanyNews(BaseModel):
    """
    公司新闻数据模型
    定义了单条公司新闻的数据结构
    """
    ticker: str  # 股票代码：新闻涉及的公司股票标识符
    title: str  # 新闻标题：新闻的主要标题
    author: str  # 作者：新闻的撰写者或记者姓名
    source: str  # 新闻来源：发布新闻的媒体机构或网站
    date: str  # 发布日期：新闻发布的日期时间
    url: str  # 新闻链接：新闻的完整URL地址
    sentiment: str | None = None  # 情感倾向：新闻的情感分析结果（正面/负面/中性）


class CompanyNewsResponse(BaseModel):
    """
    公司新闻响应模型
    包含新闻数据列表
    """
    news: list[CompanyNews]  # 新闻列表：包含多条公司相关新闻


class CompanyFacts(BaseModel):
    """
    公司基本信息模型
    包含公司的基本信息数据
    """
    ticker: str  # 股票代码：公司股票的唯一标识符
    name: str  # 公司名称：公司的完整法定名称
    cik: str | None = None  # CIK号码：SEC中央索引键，用于识别公司的唯一编号
    industry: str | None = None  # 行业：公司所属的具体行业分类
    sector: str | None = None  # 板块：公司所属的大类板块（如科技、金融等）
    category: str | None = None  # 类别：公司的业务类别或分类
    exchange: str | None = None  # 交易所：股票交易的证券交易所（如NYSE、NASDAQ）
    is_active: bool | None = None  # 是否活跃：公司股票是否仍在交易
    listing_date: str | None = None  # 上市日期：股票首次公开交易的日期
    location: str | None = None  # 公司位置：公司总部所在地
    market_cap: float | None = None  # 市值：公司的总市值
    number_of_employees: int | None = None  # 员工数量：公司的总员工人数
    sec_filings_url: str | None = None  # SEC文件链接：公司SEC文件的查询链接
    sic_code: str | None = None  # SIC代码：标准行业分类代码
    sic_industry: str | None = None  # SIC行业：基于SIC代码的行业分类
    sic_sector: str | None = None  # SIC板块：基于SIC代码的板块分类
    website_url: str | None = None  # 公司网站：公司官方网站地址
    weighted_average_shares: int | None = None  # 加权平均股数：用于计算每股指标的加权平均流通股数


class CompanyFactsResponse(BaseModel):
    """
    公司基本信息响应模型
    包含公司基本信息数据
    """
    company_facts: CompanyFacts  # 公司基本信息：包含完整的公司基础数据


class Position(BaseModel):
    """
    持仓位置模型
    定义了单个持仓的数据结构
    """
    cash: float = 0.0  # 现金：该持仓对应的现金金额
    shares: int = 0  # 持股数量：持有的股票数量
    ticker: str  # 股票代码：持仓股票的标识符


class Portfolio(BaseModel):
    """
    投资组合模型
    包含所有持仓位置和总现金
    """
    positions: dict[str, Position]  # 持仓字典：股票代码到持仓位置的映射
    total_cash: float = 0.0  # 总现金：投资组合中的可用现金总额


class AnalystSignal(BaseModel):
    """
    分析师信号模型
    定义了分析师产生的交易信号数据结构
    """
    signal: str | None = None  # 交易信号：买入/卖出/持有等交易建议
    confidence: float | None = None  # 信心度：对交易信号的信心程度（0-1之间）
    reasoning: dict | str | None = None  # 推理过程：产生该信号的分析逻辑和理由
    max_position_size: float | None = None  # 最大仓位：风险管理建议的最大持仓比例


class TickerAnalysis(BaseModel):
    """
    股票分析模型
    包含某只股票的所有分析师信号
    """
    ticker: str  # 股票代码：被分析股票的标识符
    analyst_signals: dict[str, AnalystSignal]  # 分析师信号字典：分析师名称到信号的映射


class AgentStateData(BaseModel):
    """
    智能体状态数据模型
    定义了智能体的状态数据结构
    """
    tickers: list[str]  # 股票列表：智能体关注或分析的股票代码列表
    portfolio: Portfolio  # 投资组合：智能体管理的投资组合
    start_date: str  # 开始日期：分析或交易的起始日期
    end_date: str  # 结束日期：分析或交易的结束日期
    ticker_analyses: dict[str, TickerAnalysis]  # 股票分析字典：股票代码到分析结果的映射


class AgentStateMetadata(BaseModel):
    """
    智能体状态元数据模型
    定义了智能体状态的元数据结构
    """
    show_reasoning: bool = False  # 显示推理：是否显示智能体的推理过程
    model_config = {"extra": "allow"}  # 模型配置：允许添加额外的元数据字段
