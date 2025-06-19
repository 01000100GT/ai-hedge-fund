"""
LLM模型相关的类和函数定义文件
该文件包含了与大语言模型(LLM)交互相关的所有类定义和工具函数，
包括模型提供商枚举、模型配置类以及获取模型实例的函数等。
"""

import os
import json
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from enum import Enum
from pydantic import BaseModel
from typing import Tuple, Optional, List
from dataclasses import dataclass
from colorama import Fore, Style
from pathlib import Path


class ModelProvider(str, Enum):
    """
    支持的LLM提供商枚举类
    定义了系统支持的所有AI模型服务提供商
    """

    ANTHROPIC = "Anthropic"
    DEEPSEEK = "DeepSeek"
    GEMINI = "Gemini"
    GROQ = "Groq"
    OPENAI = "OpenAI"
    OLLAMA = "Ollama"
    OPENAI_COMPATIBLE = "OpenAICompatible"


@dataclass
class LLMModel:
    """
    LLM模型配置类
    表示一个LLM模型的配置信息，包含显示名称、模型名称和提供商信息
    """

    display_name: str
    model_name: str
    provider: ModelProvider

    def to_choice_tuple(self) -> Tuple[str, str, str]:
        """将模型信息转换为questionary选项所需的格式"""
        return (self.display_name, self.model_name, self.provider.value)

    def is_custom(self) -> bool:
        """Check if the model is a Gemini model"""
        return self.model_name == "-"

    def has_json_mode(self) -> bool:
        """检查模型是否支持JSON模式输出"""
        if self.is_deepseek() or self.is_gemini():
            return False
        # Only certain Ollama models support JSON mode
        if self.is_ollama():
            return "llama3" in self.model_name or "neural-chat" in self.model_name
        return True

    def is_deepseek(self) -> bool:
        """检查是否为DeepSeek模型"""
        return self.model_name.startswith("deepseek")

    def is_gemini(self) -> bool:
        """检查是否为Gemini模型"""
        return self.model_name.startswith("gemini")

    def is_ollama(self) -> bool:
        """检查是否为Ollama模型"""
        return self.provider == ModelProvider.OLLAMA


# Load models from JSON file
def load_models_from_json(json_path: str) -> List[LLMModel]:
    """Load models from a JSON file"""
    with open(json_path, "r") as f:
        models_data = json.load(f)

    models = []
    for model_data in models_data:
        # Convert string provider to ModelProvider enum
        provider_enum = ModelProvider(model_data["provider"])
        models.append(LLMModel(display_name=model_data["display_name"], model_name=model_data["model_name"], provider=provider_enum))
    return models


# Get the path to the JSON files
current_dir = Path(__file__).parent
models_json_path = current_dir / "api_models.json"
ollama_models_json_path = current_dir / "ollama_models.json"

# Load available models from JSON
AVAILABLE_MODELS = load_models_from_json(str(models_json_path))

# Load Ollama models from JSON
OLLAMA_MODELS = load_models_from_json(str(ollama_models_json_path))

# Create LLM_ORDER in the format expected by the UI
LLM_ORDER = [model.to_choice_tuple() for model in AVAILABLE_MODELS]

# Create Ollama LLM_ORDER separately
OLLAMA_LLM_ORDER = [model.to_choice_tuple() for model in OLLAMA_MODELS]


def get_model_info(model_name: str, model_provider: str) -> LLMModel | None:
    """Get model information by model_name"""
    if model_name == "openai_compatible_custom":
        return LLMModel(display_name="OpenAI Compatible (Custom Endpoint via env vars)", model_name="custom_openai_compatible_model", provider=ModelProvider.OPENAI_COMPATIBLE)
    all_models = AVAILABLE_MODELS + OLLAMA_MODELS
    return next((model for model in all_models if model.model_name == model_name and model.provider == model_provider), None)


def get_models_list():
    """Get the list of models for API responses."""
    return [{"display_name": model.display_name, "model_name": model.model_name, "provider": model.provider.value} for model in AVAILABLE_MODELS]


def get_model(model_name: str, model_provider: ModelProvider) -> ChatOpenAI | ChatGroq | ChatOllama | None:
    """
    根据模型名称和提供商获取模型实例
    Args:
        model_name: 模型名称
        model_provider: 模型提供商
    Returns:
        返回对应的模型实例，如果创建失败则返回None
    """
    if model_provider == ModelProvider.GROQ:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            # Print error to console
            print(f"API Key Error: Please make sure GROQ_API_KEY is set in your .env file.")
            raise ValueError("Groq API key not found.  Please make sure GROQ_API_KEY is set in your .env file.")
        return ChatGroq(model=model_name, api_key=api_key)
    elif model_provider == ModelProvider.OPENAI:
        # Get and validate API key
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        if not api_key:
            # Print error to console
            print(f"API Key Error: Please make sure OPENAI_API_KEY is set in your .env file.")
            raise ValueError("OpenAI API key not found.  Please make sure OPENAI_API_KEY is set in your .env file.")
        return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
    elif model_provider == ModelProvider.ANTHROPIC:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure ANTHROPIC_API_KEY is set in your .env file.")
            raise ValueError("Anthropic API key not found.  Please make sure ANTHROPIC_API_KEY is set in your .env file.")
        return ChatAnthropic(model=model_name, api_key=api_key)
    elif model_provider == ModelProvider.DEEPSEEK:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure DEEPSEEK_API_KEY is set in your .env file.")
            raise ValueError("DeepSeek API key not found.  Please make sure DEEPSEEK_API_KEY is set in your .env file.")
        return ChatDeepSeek(model=model_name, api_key=api_key)
    elif model_provider == ModelProvider.GEMINI:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure GOOGLE_API_KEY is set in your .env file.")
            raise ValueError("Google API key not found.  Please make sure GOOGLE_API_KEY is set in your .env file.")
        return ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
    elif model_provider == ModelProvider.OLLAMA:
        # For Ollama, we use a base URL instead of an API key
        # Check if OLLAMA_HOST is set (for Docker on macOS)
        ollama_host = os.getenv("OLLAMA_HOST", "localhost")
        base_url = os.getenv("OLLAMA_BASE_URL", f"http://{ollama_host}:11434")
        return ChatOllama(
            model=model_name,
            base_url=base_url,
        )
    elif model_provider == ModelProvider.OPENAI_COMPATIBLE:
        # --- Handling OpenAI Compatible Endpoint ---
        # 从环境变量获取基础 URL (必需)
        api_base = os.getenv("OPENAI_API_BASE")
        if not api_base:
            # 如果未设置 OPENAI_API_BASE，打印错误并抛出异常
            print(f"{Fore.RED}Configuration Error: OPENAI_API_BASE environment variable is not set for OpenAI Compatible endpoint.{Style.RESET_ALL}")
            raise ValueError("OPENAI_API_BASE must be set in .env for OpenAI Compatible endpoint.")

        # 从环境变量获取 API Key (可选, 取决于端点是否需要认证)
        # 为方便起见，可以复用 OPENAI_API_KEY，或者定义一个特定的变量如 OPENAI_COMPATIBLE_API_KEY
        api_key = os.getenv("OPENAI_API_KEY")

        # 从环境变量获取实际的模型名称 (可选但推荐)
        # 如果未设置 OPENAI_COMPATIBLE_MODEL_NAME，则使用传入的 model_name (即占位符 "custom_openai_compatible_model")
        actual_model_name = os.getenv("OPENAI_COMPATIBLE_MODEL_NAME", model_name)

        # 打印使用的配置信息
        print(f"{Fore.YELLOW}Using OpenAI Compatible Endpoint:")
        print(f"  Base URL: {api_base}")
        print(f"  Model Name: {actual_model_name}")
        print(f"  API Key: {'Provided' if api_key else 'Not Provided'}{Style.RESET_ALL}")

        # 使用获取到的配置初始化 ChatOpenAI
        try:
            # 注意: 根据 langchain-openai 版本，参数可能是 base_url/api_key 或 openai_api_base/openai_api_key
            # 假设使用较新版本，参数为 base_url 和 api_key
            llm_params = {
                "model": actual_model_name,
                "base_url": api_base,
            }
            # 如果 API Key 存在，则添加到参数中
            if api_key:
                llm_params["api_key"] = api_key

            # 创建 ChatOpenAI 实例
            return ChatOpenAI(**llm_params)
        except Exception as e:
            # 如果初始化失败，打印错误并重新抛出异常
            print(f"{Fore.RED}Error initializing ChatOpenAI for compatible endpoint: {e}{Style.RESET_ALL}")
            raise

    # 如果没有匹配的 provider，可以返回 None 或抛出错误
    print(f"{Fore.RED}Error: Unsupported model provider '{model_provider}'.{Style.RESET_ALL}")
    return None
