"""
对冲基金路由模块
处理对冲基金相关的API端点，包括运行模拟和获取结果
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import asyncio

from app.backend.models.schemas import ErrorResponse, HedgeFundRequest
from app.backend.models.events import StartEvent, ProgressUpdateEvent, ErrorEvent, CompleteEvent
from app.backend.services.graph import create_graph, parse_hedge_fund_response, run_graph_async
from app.backend.services.portfolio import create_portfolio
from src.utils.progress import progress
from src.utils.analysts import get_agents_list
from src.llm.models import get_models_list

# 创建路由器实例
router = APIRouter(prefix="/hedge-fund")


@router.post(
    path="/run",
    responses={
        200: {"description": "成功响应并返回流式更新"},
        400: {"model": ErrorResponse, "description": "无效的请求参数"},
        500: {"model": ErrorResponse, "description": "服务器内部错误"},
    },
)
async def run_hedge_fund(request: HedgeFundRequest):
    """
    运行对冲基金模拟

    处理对冲基金模拟请求，返回SSE流式响应，包含进度更新和最终结果
    """
    try:
        # 创建投资组合
        portfolio = create_portfolio(request.initial_cash, request.margin_requirement, request.tickers)

        # 构建代理图
        graph = create_graph(request.selected_agents)
        graph = graph.compile()

        # 记录系统状态更新
        progress.update_status("system", None, "准备运行对冲基金模拟")

        # 将model_provider转换为字符串（如果是枚举）
        model_provider = request.model_provider
        if hasattr(model_provider, "value"):
            model_provider = model_provider.value

        # 设置流式响应
        async def event_generator():
            # 进度更新队列
            progress_queue = asyncio.Queue()

            # 简单的处理器用于将更新添加到队列
            def progress_handler(agent_name, ticker, status, analysis, timestamp):
                event = ProgressUpdateEvent(agent=agent_name, ticker=ticker, status=status, timestamp=timestamp, analysis=analysis)
                progress_queue.put_nowait(event)

            # 在进度跟踪器中注册处理器
            progress.register_handler(progress_handler)

            try:
                # 在后台任务中启动图执行
                run_task = asyncio.create_task(
                    run_graph_async(
                        graph=graph,
                        portfolio=portfolio,
                        tickers=request.tickers,
                        start_date=request.start_date,
                        end_date=request.end_date,
                        model_name=request.model_name,
                        model_provider=model_provider,
                        request=request,  # Pass the full request for agent-specific model access
                    )
                )
                # 发送初始消息
                yield StartEvent().to_sse()

                # 流式传输进度更新直到run_task完成
                while not run_task.done():
                    # 获取进度更新或等待
                    try:
                        event = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                        yield event.to_sse()
                    except asyncio.TimeoutError:
                        # 继续循环
                        pass

                # 获取最终结果
                result = run_task.result()

                if not result or not result.get("messages"):
                    yield ErrorEvent(message="生成对冲基金决策失败").to_sse()
                    return

                # 发送最终结果
                final_data = CompleteEvent(
                    data={
                        "decisions": parse_hedge_fund_response(result.get("messages", [])[-1].content),
                        "analyst_signals": result.get("data", {}).get("analyst_signals", {}),
                    }
                )
                yield final_data.to_sse()

            finally:
                # 清理
                progress.unregister_handler(progress_handler)
                if "run_task" in locals() and not run_task.done():
                    run_task.cancel()

        # 返回流式响应
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")


@router.get(
    path="/agents",
    responses={
        200: {"description": "List of available agents"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_agents():
    """Get the list of available agents."""
    try:
        return {"agents": get_agents_list()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agents: {str(e)}")


@router.get(
    path="/language-models",
    responses={
        200: {"description": "List of available LLMs"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_language_models():
    """Get the list of available models."""
    try:
        return {"models": get_models_list()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve models: {str(e)}")
