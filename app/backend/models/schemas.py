"""
数据模型模块
定义API使用的请求和响应数据模型
"""
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import List, Optional
from src.llm.models import ModelProvider


class HedgeFundResponse(BaseModel):
    """
    对冲基金响应模型
    包含交易决策和分析师信号
    """
    decisions: dict
    analyst_signals: dict


class ErrorResponse(BaseModel):
    """
    错误响应模型
    包含错误消息和可选的错误详情
    """
    message: str
    error: str | None = None


class HedgeFundRequest(BaseModel):
    """
    对冲基金请求模型
    包含运行对冲基金模拟所需的所有参数
    """
    tickers: List[str]  # 股票代码列表
    selected_agents: List[str]  # 选定的代理列表
    end_date: Optional[str] = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))  # 结束日期
    start_date: Optional[str] = None  # 开始日期
    model_name: str = "gpt-4o"  # 使用的模型名称
    model_provider: ModelProvider = ModelProvider.OPENAI  # 模型提供者
    initial_cash: float = 100000.0  # 初始现金金额
    margin_requirement: float = 0.0  # 保证金要求

    def get_start_date(self) -> str:
        """
        计算开始日期
        如果未提供开始日期，则返回结束日期前90天
        """
        if self.start_date:
            return self.start_date
        return (datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
