"""
投资组合服务模块
提供创建和管理投资组合的功能
"""

def create_portfolio(initial_cash: float, margin_requirement: float, tickers: list[str]) -> dict:
    """
    创建新的投资组合
    
    参数:
        initial_cash: 初始现金金额
        margin_requirement: 保证金要求
        tickers: 股票代码列表
        
    返回:
        包含投资组合初始状态的字典
    """
    return {
        "cash": initial_cash,  # 初始现金金额
        "margin_requirement": margin_requirement,  # 保证金要求
        "margin_used": 0.0,  # 所有空头头寸使用的总保证金
        "positions": {
            ticker: {
                "long": 0,  # 持有的多头股票数量
                "short": 0,  # 持有的空头股票数量
                "long_cost_basis": 0.0,  # 多头头寸的平均成本基础
                "short_cost_basis": 0.0,  # 做空股票时的平均价格
                "short_margin_used": 0.0,  # 该股票空头使用的保证金金额
            }
            for ticker in tickers
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,  # 多头头寸已实现收益
                "short": 0.0,  # 空头头寸已实现收益
            }
            for ticker in tickers
        },
    }