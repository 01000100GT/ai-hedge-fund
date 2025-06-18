[tool.poetry]
name = "ai-hedge-fund"
version = "0.1.0"
description = "An AI-powered hedge fund that uses multiple agents to make trading decisions"
authors = ["Your Name <your.email@example.com>"]
readme = "README.md"
packages = [
    { include = "src", from = "." },
    { include = "app", from = "." }
]

[tool.poetry.dependencies]
python = "3.11.*"  # Python解释器版本要求
langchain = "0.3.0"  # LangChain框架核心包，用于构建AI应用
langchain-anthropic = "0.3.5"  # Anthropic语言模型集成
langchain-groq = "0.2.3"  # Groq语言模型集成
langchain-openai = "^0.3.5"  # OpenAI语言模型集成
langchain-deepseek = "^0.1.2"  # DeepSeek语言模型集成
langchain-ollama = "^0.2.0"  # Ollama本地语言模型集成
langgraph = "0.2.56"  # LangChain的图形编排工具
pandas = "^2.1.0"  # 数据分析和处理库
numpy = "^1.24.0"  # 科学计算库
python-dotenv = "1.0.0"  # 环境变量管理工具
matplotlib = "^3.9.2"  # 数据可视化库
tabulate = "^0.9.0"  # 表格格式化工具
colorama = "^0.4.6"  # 终端文字颜色处理
questionary = "^2.1.0"  # 命令行交互工具
rich = "^13.9.4"  # 终端富文本和格式化输出
langchain-google-genai = "^2.0.11"  # Google AI语言模型集成

# 后端依赖
fastapi = {extras = ["standard"], version = "^0.104.0"}  # FastAPI Web框架
fastapi-cli = "^0.0.7"  # FastAPI命令行工具
pydantic = "^2.4.2"  # 数据验证库
httpx = "^0.27.0"  # 异步HTTP客户端
sqlalchemy = "^2.0.22"  # ORM数据库工具
alembic = "^1.12.0"  # 数据库迁移工具

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"  # 单元测试框架
black = "^23.7.0"  # 代码格式化工具
isort = "^5.12.0"  # 导入语句排序工具
flake8 = "^6.1.0"  # 代码风格检查工具

[[tool.poetry.source]]
name = "mirrors"
url = "https://mirrors.aliyun.com/pypi/simple/"
priority = "primary"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 420
target-version = ['py311']
include = '\.pyi?$'

[tool.isort]
profile = "black"
force_alphabetical_sort_within_sections = true 