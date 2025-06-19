"""
事件模型模块
定义服务器发送事件(SSE)的数据模型
"""

from typing import Dict, Optional, Any, Literal
from pydantic import BaseModel


class BaseEvent(BaseModel):
    """
    所有服务器发送事件的基类
    提供基本的事件类型和SSE格式转换功能
    """

    type: str

    def to_sse(self) -> str:
        """转换为服务器发送事件格式"""
        event_type = self.type.lower()
        return f"event: {event_type}\ndata: {self.model_dump_json()}\n\n"


class StartEvent(BaseEvent):
    """
    开始处理事件
    表示开始处理请求
    """

    type: Literal["start"] = "start"
    timestamp: Optional[str] = None


class ProgressUpdateEvent(BaseEvent):
    """
    进度更新事件
    包含代理的进度更新信息
    """

    type: Literal["progress"] = "progress"
    agent: str
    ticker: Optional[str] = None
    status: str
    timestamp: Optional[str] = None
    analysis: Optional[str] = None


class ErrorEvent(BaseEvent):
    """
    错误事件
    表示处理过程中发生的错误
    """

    type: Literal["error"] = "error"
    message: str
    timestamp: Optional[str] = None


class CompleteEvent(BaseEvent):
    """
    完成事件
    包含成功完成处理的结果数据
    """

    type: Literal["complete"] = "complete"
    data: Dict[str, Any]
    timestamp: Optional[str] = None
