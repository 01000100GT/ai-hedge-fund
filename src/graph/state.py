"""
图状态管理模块
该模块定义了智能体(Agent)的状态管理相关类和函数，
包括消息、数据和元数据的管理，以及状态展示功能。
"""

from typing_extensions import Annotated, Sequence, TypedDict

import operator
from langchain_core.messages import BaseMessage


import json


def merge_dicts(a: dict[str, any], b: dict[str, any]) -> dict[str, any]:
    """
    合并两个字典
    Args:
        a: 第一个字典
        b: 第二个字典
    Returns:
        返回合并后的新字典
    """
    return {**a, **b}


# Define agent state
class AgentState(TypedDict):
    """
    智能体状态类
    定义了智能体的状态结构，包含消息历史、数据和元数据
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict[str, any], merge_dicts]
    metadata: Annotated[dict[str, any], merge_dicts]


def show_agent_reasoning(output, agent_name):
    """
    展示智能体的推理过程
    Args:
        output: 输出内容
        agent_name: 智能体名称
    """
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")

    def convert_to_serializable(obj):
        """
        将对象转换为可序列化的格式
        Args:
            obj: 需要转换的对象
        Returns:
            返回可序列化的对象
        """
        if hasattr(obj, "to_dict"):  # Handle Pandas Series/DataFrame
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):  # Handle custom objects
            return obj.__dict__
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)  # Fallback to string representation

    if isinstance(output, (dict, list)):
        # Convert the output to JSON-serializable format
        serializable_output = convert_to_serializable(output)
        print(json.dumps(serializable_output, indent=2))
    else:
        try:
            # Parse the string as JSON and pretty print it
            parsed_output = json.loads(output)
            print(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            # Fallback to original string if not valid JSON
            print(output)

    print("=" * 48)
