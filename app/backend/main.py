"""
AI对冲基金后端API主程序
提供FastAPI应用程序的主要入口点和配置
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.routes import api_router

# 创建FastAPI应用实例
app = FastAPI(title="AI Hedge Fund API", description="Backend API for AI Hedge Fund", version="0.1.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routes
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn

    # 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=8000)
