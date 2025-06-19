[project] # 项目配置
name = "ai-hedge-fund" # 项目名称
version = "0.1.0" # 项目版本
description = "An AI-powered hedge fund that uses multiple agents to make trading decisions" # 项目描述
authors = [ # 作者信息
    {name = "Your Name", email = "your.email@example.com"}
]
readme = "README.md" # README 文件
requires-python = ">=3.11,<3.12" # Python 版本要求
dependencies = [ # 项目依赖库
    "langchain==0.3.0", # LangChain 库
    "langchain-anthropic==0.3.5", # LangChain Anthropic 集成
    "langchain-groq==0.2.3", # LangChain Groq 集成
    "langchain-openai>=0.3.5", # LangChain OpenAI 集成
    "langchain-deepseek>=0.1.2", # LangChain DeepSeek 集成
    "langchain-ollama>=0.2.0", # LangChain Ollama 集成
    "langgraph==0.2.56", # LangGraph 库
    "pandas>=2.1.0", # Pandas 数据处理库
    "numpy>=1.24.0", # NumPy 数值计算库
    "python-dotenv==1.0.0", # Python Dotenv 库，用于加载环境变量
    "matplotlib>=3.9.2", # Matplotlib 绘图库
    "tabulate>=0.9.0", # Tabulate 库，用于格式化表格输出
    "colorama>=0.4.6", # Colorama 库，用于彩色终端输出
    "questionary>=2.1.0", # Questionary 库，用于交互式命令行提示
    "rich>=13.9.4", # Rich 库，用于富文本和美观的终端输出
    "langchain-google-genai>=2.0.11", # LangChain Google GenAI 集成
    "fastapi[standard]>=0.104.0", # FastAPI 框架，带有标准依赖
    "fastapi-cli>=0.0.7", # FastAPI 命令行工具
    "pydantic>=2.4.2", # Pydantic 数据验证库
    "httpx>=0.27.0", # HTTpx HTTP 客户端
    "sqlalchemy>=2.0.22", # SQLAlchemy ORM 库
    "alembic>=1.12.0", # Alembic 数据库迁移工具
]

[project.optional-dependencies] # 可选依赖
dev = [ # 开发依赖
    "pytest>=7.4.0", # Pytest 测试框架
    "black>=23.7.0", # Black 代码格式化工具
    "isort>=5.12.0", # Isort 导入排序工具
    "flake8>=6.1.0", # Flake8 代码风格检查工具
]

[build-system] # 构建系统配置
requires = ["hatchling"] # 构建系统所需依赖
build-backend = "hatchling.build" # 构建后端

[[tool.uv.index]] # UV 工具索引配置
url = "https://pypi.tuna.tsinghua.edu.cn/simple" # PyPI 镜像源地址
default = true # 设置为默认索引

[tool.hatch.build.targets.wheel] # Hatch 构建工具的 wheel 目标配置
packages = ["src", "app"] # 要包含在 wheel 中的包

[tool.black] # Black 代码格式化工具配置
line-length = 420 # 行长度限制
target-version = ['py311'] # 目标 Python 版本
include = '\.pyi?$' # 包含的文件模式

[tool.isort] # Isort 导入排序工具配置
profile = "black" # 使用 Black 配置文件
force_alphabetical_sort_within_sections = true # 在每个 section 内强制按字母顺序排序 