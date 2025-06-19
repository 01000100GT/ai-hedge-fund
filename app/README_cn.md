# AI 对冲基金 [WIP] 🚧

本项目目前正在开发中。要跟踪进度，请在此处获取更新：[https://x.com/virattt](https://x.com/virattt)。

AI 对冲基金应用程序是一个包含前端和后端组件的完整系统，使您可以通过计算机上的 Web 界面运行 AI 驱动的对冲基金交易系统。

<img width="1692" alt="Screenshot 2025-05-15 at 8 57 56 PM" src="https://github.com/user-attachments/assets/2173fa5b-1029-49dd-8b04-d7583616de1b" />


## 概述

AI 对冲基金包括：

- **后端**：一个 FastAPI 应用程序，提供 REST API 以运行对冲基金交易系统和回测器
- **前端**：一个 React/Vite 应用程序，提供用户友好的界面来可视化和控制对冲基金操作

## 目录

- [🚀 快速开始（非技术用户）](#-快速开始非技术用户)
  - [选项 1：使用一行 Shell 脚本（推荐）](#选项-1使用一行-shell-脚本推荐)
  - [选项 2：使用 npm（替代方案）](#选项-2使用-npm替代方案)
- [🛠️ 手动设置（开发者）](#️-手动设置开发者)
  - [先决条件](#先决条件)
  - [安装](#安装)
  - [运行应用程序](#运行应用程序)
- [详细文档](#详细文档)
- [免责声明](#免责声明)

## 🚀 快速开始（非技术用户）

**一行设置和运行命令：**

### 选项 1：使用一行 Shell 脚本（推荐）

#### 适用于 Mac/Linux：
```bash
./run.sh
```

如果您遇到"权限被拒绝"错误，请先运行此命令：
```bash
chmod +x run.sh && ./run.sh
```

或者，您也可以运行：
```bash
bash run.sh
```

#### 适用于 Windows：
```cmd
run.bat
```

### 选项 2：使用 npm（替代方案）
```bash
cd app && npm install && npm run setup
```

**就这么简单！**这些脚本将：
1. 检查所需的依赖项（Node.js、Python、Poetry）
2. 自动安装所有依赖项
3. 启动前端和后端服务
4. **自动在您的 Web 浏览器中打开**应用程序

**要求：**
- [Node.js](https://nodejs.org/) (包括 npm)
- [Python 3](https://python.org/)
- [Poetry](https://python-poetry.org/)

**运行后，您可以访问：**
- 前端（Web 界面）：http://localhost:5173
- 后端 API：http://localhost:8000
- API 文档：http://localhost:8000/docs

---

## 🛠️ 手动设置（开发者）

如果您更喜欢手动设置每个组件或需要更多控制：

### 先决条件

- 前端需要 Node.js 和 npm
- 后端需要 Python 3.8+ 和 Poetry

### 安装

1. 克隆仓库：
```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

2. 设置您的环境变量：
```bash
# 为您的 API 密钥创建 .env 文件（在根目录中）
cp .env.example .env
```

3. 编辑 .env 文件以添加您的 API 密钥：
```bash
# 用于运行 OpenAI 托管的 LLM（gpt-4o, gpt-4o-mini 等）
OPENAI_API_KEY=your-openai-api-key

# 用于运行 Groq 托管的 LLM（deepseek, llama3 等）
GROQ_API_KEY=your-groq-api-key

# 用于获取对冲基金所需的金融数据
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
```

4. 安装 Poetry（如果尚未安装）：
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

5. 安装根项目依赖项：
```bash
# 从根目录
poetry install
```

6. 安装后端应用程序依赖项：
```bash
# 导航到后端目录
cd app/backend
pip install -r requirements.txt  # 如果有 requirements.txt 文件
# 或者
poetry install  # 如果后端目录中有 pyproject.toml
```

7. 安装前端应用程序依赖项：
```bash
cd app/frontend
npm install  # 或 pnpm install 或 yarn install
```

### 运行应用程序

1. 启动后端服务器：
```bash
# 在一个终端中，从后端目录
cd app/backend
poetry run uvicorn main:app --reload
```

2. 启动前端应用程序：
```bash
# 在另一个终端中，从前端目录
cd app/frontend
npm run dev
```

您现在可以访问：
- 前端（Web 界面）：http://localhost:5173
- 后端 API：http://localhost:8000
- API 文档：http://localhost:8000/docs

## 详细文档

有关更多详细信息：
- [后端文档](./backend/README.md)
- [前端文档](./frontend/README.md)

## 免责声明

本项目仅用于**教育和研究目的**。

- 不适用于实际交易或投资
- 不提供任何保证或担保
- 创建者不对任何财务损失承担责任
- 投资决策请咨询财务顾问

使用本软件即表示您同意仅将其用于学习目的。 