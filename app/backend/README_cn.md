# AI 对冲基金 - 后端 [WIP] 🚧
本项目目前正在开发中。要跟踪进度，请在此处获取更新：[https://x.com/virattt](https://x.com/virattt)。

这是 AI 对冲基金项目的后端服务器。它提供了一个简单的 REST API 来与 AI 对冲基金系统进行交互，使您能够通过 Web 界面运行对冲基金。

## 概述

该后端项目是一个 FastAPI 应用程序，作为 AI 对冲基金系统的服务器端组件。它公开了用于运行对冲基金交易系统和回测器的端点。

该后端旨在与未来的前端应用程序配合使用，该应用程序将允许用户通过浏览器与 AI 对冲基金系统进行交互。

## 安装

### 使用 Poetry

1. 克隆仓库：
```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

2. 安装 Poetry（如果尚未安装）：
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. 安装依赖：
```bash
# 从根目录
poetry install
```

4. 设置您的环境变量：
```bash
# 为您的 API 密钥创建 .env 文件（在根目录中）
cp .env.example .env
```

5. 编辑 .env 文件以添加您的 API 密钥：
```bash
# 用于运行 OpenAI 托管的 LLM（gpt-4o, gpt-4o-mini 等）
OPENAI_API_KEY=your-openai-api-key

# 用于运行 Groq 托管的 LLM（deepseek, llama3 等）
GROQ_API_KEY=your-groq-api-key

# 用于获取对冲基金所需的金融数据
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
```

## 运行服务器

要运行开发服务器：

```bash
# 导航到后端目录
cd app/backend

# 使用 uvicorn 启动 FastAPI 服务器
poetry run uvicorn main:app --reload
```

这将启动启用热重载的 FastAPI 服务器。

API 将在以下地址可用：
- API 端点：http://localhost:8000
- API 文档：http://localhost:8000/docs

## API 端点

- `POST /hedge-fund/run`：使用指定参数运行 AI 对冲基金
- `GET /ping`：用于测试服务器连接的简单端点

## 项目结构

```
app/backend/
├── api/                      # API 层（未来扩展）
├── models/                   # 领域模型
│   ├── __init__.py
│   └── schemas.py            # Pydantic 模式定义
├── routes/                   # API 路由
│   ├── __init__.py           # 路由器注册表
│   ├── hedge_fund.py         # 对冲基金端点
│   └── health.py             # 健康检查端点
├── services/                 # 业务逻辑
│   ├── graph.py              # 代理图功能
│   └── portfolio.py          # 投资组合管理
├── __init__.py               # 包初始化
└── main.py                   # FastAPI 应用程序入口点
```

## 免责声明

本项目仅用于**教育和研究目的**。

- 不适用于实际交易或投资
- 不提供任何保证或担保
- 创建者不对任何财务损失承担责任
- 投资决策请咨询财务顾问

使用本软件即表示您同意仅将其用于学习目的。 