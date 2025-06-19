"""LLM（大语言模型）相关的辅助函数"""

import json
from typing import TypeVar, Type, Optional, Any
from pydantic import BaseModel
from utils.progress import progress

T = TypeVar('T', bound=BaseModel)

def call_llm(
    prompt: Any,
    model_name: str,
    model_provider: str,
    pydantic_model: Type[T],
    agent_name: Optional[str] = None,
    max_retries: int = 3,
    default_factory = None
) -> T:
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
    from llm.models import get_model, get_model_info
    
    model_info = get_model_info(model_name)
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

def create_default_response(model_class: Type[T]) -> T:
    """根据模型的字段创建安全的默认响应。"""
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

def extract_json_from_response(content: str) -> Optional[dict]:
    """从markdown格式的响应中提取JSON。"""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7:]  # 跳过```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"从响应中提取JSON时出错: {e}")
    return None
