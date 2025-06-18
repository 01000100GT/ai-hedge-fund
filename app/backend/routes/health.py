"""
健康检查路由模块
提供API的健康检查和心跳检测端点
"""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio
import json

router = APIRouter()


@router.get("/")
async def root():
    """
    API根路径处理器
    返回欢迎消息
    """
    return {"message": "Welcome to AI Hedge Fund API"}


@router.get("/ping")
async def ping():
    """
    心跳检测处理器
    返回SSE流式响应，每秒发送一次ping消息，共发送5次
    """
    async def event_generator():
        for i in range(5):
            # 为每个ping创建JSON对象
            data = {"ping": f"ping {i+1}/5", "timestamp": i + 1}

            # 格式化为SSE
            yield f"data: {json.dumps(data)}\n\n"

            # 等待1秒
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
