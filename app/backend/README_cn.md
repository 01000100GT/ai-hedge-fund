# AI 对冲基金 - 后端 [进行中] 🚧
本项目目前正在进行中。要跟踪进度，请在[这里](https://x.com/virattt)获取更新。

这是AI对冲基金项目的后端服务器。它提供了一个简单的REST API与AI对冲基金系统交互，使您能够通过Web界面运行对冲基金。

## 概述

这个后端项目是一个FastAPI应用程序，作为AI对冲基金系统的服务器端组件。它公开了用于运行对冲基金交易系统和回测器的端点。

这个后端设计用于与未来的前端应用程序配合使用，允许用户通过浏览器与AI对冲基金系统交互。

## 安装

### 使用Poetry

1. 克隆仓库：
```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

2. 安装Poetry（如果尚未安装）：
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. 安装依赖：
```bash
# 从根目录
poetry install
```

4. 设置环境变量：
```bash
# 为API密钥创建.env文件（在根目录中）
cp .env.example .env
```

5. 编辑.env文件添加您的API密钥：
```bash
# 用于运行由openai托管的LLM（gpt-4o, gpt-4o-mini等）
OPENAI_API_KEY=your-openai-api-key

# 用于运行由groq托管的LLM（deepseek, llama3等）
GROQ_API_KEY=your-groq-api-key

# 用于获取为对冲基金提供动力的金融数据
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
```

## 运行服务器

要运行开发服务器：

```bash
# 导航到后端目录
cd app/backend

# 使用uvicorn启动FastAPI服务器
poetry run uvicorn main:app --reload
```

这将启动启用了热重载的FastAPI服务器。

API将在以下位置可用：
- API端点：http://localhost:8000
- API文档：http://localhost:8000/docs

## API端点

- `POST /hedge-fund/run`：使用指定参数运行AI对冲基金
- `GET /ping`：测试服务器连接的简单端点

## 项目结构

```
app/backend/
├── api/                      # API层（未来扩展）
├── models/                   # 领域模型
│   ├── __init__.py
│   └── schemas.py            # Pydantic架构定义
├── routes/                   # API路由
│   ├── __init__.py           # 路由器注册
│   ├── hedge_fund.py         # 对冲基金端点
│   └── health.py             # 健康检查端点
├── services/                 # 业务逻辑
│   ├── graph.py              # 智能体图功能
│   └── portfolio.py          # 投资组合管理
├── __init__.py               # 包初始化
└── main.py                   # FastAPI应用程序入口点
```

## 免责声明

本项目**仅供教育和研究目的使用**。

- 不用于实际交易或投资
- 不提供任何保证或担保
- 创建者对财务损失不承担任何责任
- 投资决策请咨询财务顾问

使用本软件即表示您同意仅将其用于学习目的。 