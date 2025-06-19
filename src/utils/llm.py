"""LLM（大语言模型）相关的辅助函数"""

import json
from pydantic import BaseModel
from src.llm.models import get_model, get_model_info
from src.utils.progress import progress
from src.graph.state import AgentState


def call_llm(
    prompt: any,
    pydantic_model: type[BaseModel],
    agent_name: str | None = None,
    state: AgentState | None = None,
    max_retries: int = 3,
    default_factory=None,
) -> BaseModel:
    """
    使用重试逻辑调用LLM，处理支持和不支持JSON的模型。

    参数:
        prompt: 发送给LLM的提示
        model_name: 要使用的模型名称
        model_provider: 模型提供者
        pydantic_model: 用于结构化输出的Pydantic模型类
        agent_name: 用于进度更新的可选代理名称
        max_retries: 最大重试次数（默认：3）
        default_factory: 失败时创建默认响应的可选工厂函数

    返回:
        指定Pydantic模型的实例
    """

    # Extract model configuration if state is provided and agent_name is available
    if state and agent_name:
        model_name, model_provider = get_agent_model_config(state, agent_name)

    # Fallback to defaults if still not provided
    if not model_name:
        model_name = "gpt-4o"
    if not model_provider:
        model_provider = "OPENAI"

    model_info = get_model_info(model_name, model_provider)
    llm = get_model(model_name, model_provider)

    # 对于不支持JSON的模型，我们不能用使用结构化输出（也就是支持的才能用with_structured_output）
    if not (model_info and not model_info.has_json_mode()):
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )

    # 使用重试调用LLM
    for attempt in range(max_retries):
        try:
            # 调用LLM
            result = llm.invoke(prompt)

            # 对于不支持JSON的模型，我们需要手动提取和解析JSON
            if model_info and not model_info.has_json_mode():
                parsed_result = extract_json_from_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
            else:
                return result

        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")

            if attempt == max_retries - 1:
                print(f"LLM调用在{max_retries}次尝试后出错: {e}")
                # 如果提供了default_factory则使用它，否则创建基本默认值
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # 由于上面的重试逻辑，这里永远不会被执行到
    return create_default_response(pydantic_model)


def create_default_response(model_class: type[BaseModel]) -> BaseModel:
    """据模型的字段创建安全的默认响应。"""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # 对于其他类型（如Literal），尝试使用第一个允许的值
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)


def extract_json_from_response(content: str) -> dict | None:
    """从markdown格式的响应中提取JSON。"""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7 :]  # 跳过```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"从响应中提取JSON时出错: {e}")
    return None


def get_agent_model_config(state, agent_name):
    """
    Get model configuration for a specific agent from the state.
    Falls back to global model configuration if agent-specific config is not available.
    """
    request = state.get("metadata", {}).get("request")

    if agent_name == "portfolio_manager":
        # Get the model and provider from state metadata
        model_name = state.get("metadata", {}).get("model_name", "gpt-4o")
        model_provider = state.get("metadata", {}).get("model_provider", "OPENAI")
        return model_name, model_provider

    if request and hasattr(request, "get_agent_model_config"):
        # Get agent-specific model configuration
        model_name, model_provider = request.get_agent_model_config(agent_name)
        return model_name, model_provider.value if hasattr(model_provider, "value") else str(model_provider)

    # Fall back to global configuration
    model_name = state.get("metadata", {}).get("model_name", "gpt-4o")
    model_provider = state.get("metadata", {}).get("model_provider", "OPENAI")

    # Convert enum to string if necessary
    if hasattr(model_provider, "value"):
        model_provider = model_provider.value

    return model_name, model_provider
