"""
AI对冲基金后端API主程序
提供FastAPI应用程序的主要入口点和配置
"""
from fastapi import FastAPI

from app.backend.routes import api_router

# 创建FastAPI应用实例
app = FastAPI(title="AI Hedge Fund API", description="Backend API for AI Hedge Fund", version="0.1.0")

# 包含所有路由
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    # 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=8000)
