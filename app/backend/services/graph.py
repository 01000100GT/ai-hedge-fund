"""
图形工作流服务模块
管理AI对冲基金中各个代理之间的工作流程和通信
"""

import asyncio
import json
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from src.agents.portfolio_manager import portfolio_management_agent
from src.agents.risk_manager import risk_management_agent
from src.main import start
from src.utils.analysts import ANALYST_CONFIG
from src.graph.state import AgentState


def create_graph(selected_agents: list[str]) -> StateGraph:
    """
    创建包含选定代理的工作流图

    参数:
        selected_agents: 选定的分析师代理列表
    返回:
        配置好的StateGraph实例
    """
    graph = StateGraph(AgentState)
    graph.add_node("start_node", start)

    # Filter out any agents that are not in analyst.py
    selected_agents = [agent for agent in selected_agents if agent in ANALYST_CONFIG]

    # Get analyst nodes from the configuration
    analyst_nodes = {key: (f"{key}_agent", config["agent_func"]) for key, config in ANALYST_CONFIG.items()}

    # 添加选定的分析师节点
    for agent_name in selected_agents:
        node_name, node_func = analyst_nodes[agent_name]
        graph.add_node(node_name, node_func)
        graph.add_edge("start_node", node_name)

    # 始终添加风险和投资组合管理节点
    graph.add_node("risk_management_agent", risk_management_agent)
    graph.add_node("portfolio_manager", portfolio_management_agent)

    # 将选定的代理连接到风险管理
    for agent_name in selected_agents:
        node_name = analyst_nodes[agent_name][0]
        graph.add_edge(node_name, "risk_management_agent")

    # 将风险管理代理连接到投资组合管理代理
    graph.add_edge("risk_management_agent", "portfolio_manager")

    # 将投资组合管理代理连接到结束节点
    graph.add_edge("portfolio_manager", END)

    # 设置入口点为开始节点
    graph.set_entry_point("start_node")
    return graph


async def run_graph_async(graph, portfolio, tickers, start_date, end_date, model_name, model_provider, request=None):
    """Async wrapper for run_graph to work with asyncio."""
    # Use run_in_executor to run the synchronous function in a separate thread
    # so it doesn't block the event loop
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, lambda: run_graph(graph, portfolio, tickers, start_date, end_date, model_name, model_provider, request))  # Use default executor
    return result


def run_graph(
    graph: StateGraph,
    portfolio: dict,
    tickers: list[str],
    start_date: str,
    end_date: str,
    model_name: str,
    model_provider: str,
    request=None,
) -> dict:
    """
    运行工作流图并生成交易决策

    参数:
        graph: 配置好的工作流图
        portfolio: 投资组合信息
        tickers: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        model_name: 使用的模型名称
        model_provider: 模型提供者

    返回:
        包含交易决策的字典
    """
    return graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Make trading decisions based on the provided data.",
                )
            ],
            "data": {
                "tickers": tickers,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
                "analyst_signals": {},
            },
            "metadata": {
                "show_reasoning": False,
                "model_name": model_name,
                "model_provider": model_provider,
                "request": request,  # Pass the request for agent-specific model access
            },
        },
    )


def parse_hedge_fund_response(response):
    """
    解析对冲基金响应的JSON字符串

    参数:
        response: JSON格式的响应字符串
    返回:
        解析后的字典，如果解析失败则返回None
    """
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}\n响应: {repr(response)}")
        return None
    except TypeError as e:
        print(f"无效的响应类型 (预期字符串, 得到 {type(response).__name__}): {e}")
        return None
    except Exception as e:
        print(f"解析响应时发生意外错误: {e}\n响应: {repr(response)}")
        return None
