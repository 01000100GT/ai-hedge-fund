import os
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


class ModelProvider(str, Enum):
    """Enum for supported LLM providers"""

    ANTHROPIC = "Anthropic"
    DEEPSEEK = "DeepSeek"
    GEMINI = "Gemini"
    GROQ = "Groq"
    OPENAI = "OpenAI"
    OLLAMA = "Ollama"
    OPENAI_COMPATIBLE = "OpenAICompatible"


@dataclass
class LLMModel:
    """Represents an LLM model configuration"""

    display_name: str
    model_name: str
    provider: ModelProvider

    def to_choice_tuple(self) -> Tuple[str, str, str]:
        """Convert to format needed for questionary choices"""
        return (self.display_name, self.model_name, self.provider.value)

    def has_json_mode(self) -> bool:
        """Check if the model supports JSON mode"""
        if self.is_deepseek() or self.is_gemini():
            return False
        # Only certain Ollama models support JSON mode
        if self.is_ollama():
            return "llama3" in self.model_name or "neural-chat" in self.model_name
        return True

    def is_deepseek(self) -> bool:
        """Check if the model is a DeepSeek model"""
        return self.model_name.startswith("deepseek")

    def is_gemini(self) -> bool:
        """Check if the model is a Gemini model"""
        return self.model_name.startswith("gemini")

    def is_ollama(self) -> bool:
        """Check if the model is an Ollama model"""
        return self.provider == ModelProvider.OLLAMA


# Define available models
AVAILABLE_MODELS = [
    LLMModel(display_name="[anthropic] claude-3.5-haiku", model_name="claude-3-5-haiku-latest", provider=ModelProvider.ANTHROPIC),
    LLMModel(display_name="[anthropic] claude-3.5-sonnet", model_name="claude-3-5-sonnet-latest", provider=ModelProvider.ANTHROPIC),
    LLMModel(display_name="[anthropic] claude-3.7-sonnet", model_name="claude-3-7-sonnet-latest", provider=ModelProvider.ANTHROPIC),
    LLMModel(display_name="[deepseek] deepseek-r1", model_name="deepseek-reasoner", provider=ModelProvider.DEEPSEEK),
    LLMModel(display_name="[deepseek] deepseek-v3", model_name="deepseek-chat", provider=ModelProvider.DEEPSEEK),
    LLMModel(display_name="[gemini] gemini-2.0-flash", model_name="gemini-2.0-flash", provider=ModelProvider.GEMINI),
    LLMModel(display_name="[gemini] gemini-2.5-pro", model_name="gemini-2.5-pro-exp-03-25", provider=ModelProvider.GEMINI),
    LLMModel(display_name="[groq] llama-4-scout-17b", model_name="meta-llama/llama-4-scout-17b-16e-instruct", provider=ModelProvider.GROQ),
    LLMModel(display_name="[groq] llama-4-maverick-17b", model_name="meta-llama/llama-4-maverick-17b-128e-instruct", provider=ModelProvider.GROQ),
    LLMModel(display_name="[openai] gpt-4.5", model_name="gpt-4.5-preview", provider=ModelProvider.OPENAI),
    LLMModel(display_name="[openai] gpt-4o", model_name="gpt-4o", provider=ModelProvider.OPENAI),
    LLMModel(display_name="[openai] o3", model_name="o3", provider=ModelProvider.OPENAI),
    LLMModel(display_name="[openai] o4-mini", model_name="o4-mini", provider=ModelProvider.OPENAI),
]

# Define Ollama models separately
OLLAMA_MODELS = [
    LLMModel(display_name="[google] gemma3 (4B)", model_name="gemma3:4b", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[alibaba] qwen3 (4B)", model_name="qwen3:4b", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[meta] llama3.1 (8B)", model_name="llama3.1:latest", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[google] gemma3 (12B)", model_name="gemma3:12b", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[mistral] mistral-small3.1 (24B)", model_name="mistral-small3.1", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[google] gemma3 (27B)", model_name="gemma3:27b", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[alibaba] qwen3 (30B-a3B)", model_name="qwen3:30b-a3b", provider=ModelProvider.OLLAMA),
    LLMModel(display_name="[meta] llama-3.3 (70B)", model_name="llama3.3:70b-instruct-q4_0", provider=ModelProvider.OLLAMA),
]

# Create LLM_ORDER in the format expected by the UI
LLM_ORDER = [model.to_choice_tuple() for model in AVAILABLE_MODELS]

# Create Ollama LLM_ORDER separately
OLLAMA_LLM_ORDER = [model.to_choice_tuple() for model in OLLAMA_MODELS]


def get_model_info(model_name: str) -> LLMModel | None:
    """Get model information by model_name"""
    if model_name == "openai_compatible_custom":
        return LLMModel(
            display_name="OpenAI Compatible (Custom Endpoint via env vars)",
            model_name="custom_openai_compatible_model",
            provider=ModelProvider.OPENAI_COMPATIBLE
        )

    for model in AVAILABLE_MODELS:
        if model.model_name == model_name:
            return model

    for _, ollama_model_name, _ in OLLAMA_LLM_ORDER:
        if ollama_model_name == model_name:
            return LLMModel(display_name=f"[ollama] {model_name}", model_name=model_name, provider=ModelProvider.OLLAMA)

    return None


def get_model(model_name: str, model_provider: ModelProvider) -> ChatOpenAI | ChatGroq | ChatOllama | None:
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
        if not api_key:
            # Print error to console
            print(f"API Key Error: Please make sure OPENAI_API_KEY is set in your .env file.")
            raise ValueError("OpenAI API key not found.  Please make sure OPENAI_API_KEY is set in your .env file.")
        return ChatOpenAI(model=model_name, api_key=api_key)
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
