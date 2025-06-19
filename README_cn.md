# AI 对冲基金

这是一个 AI 驱动的对冲基金的概念验证。该项目的目标是探索使用 AI 进行交易决策。本项目仅用于**教育**目的，不适用于实际交易或投资。

该系统包含多个协同工作的代理：

1. Aswath Damodaran 代理 - 估值院长，专注于故事、数据和严谨的估值
2. Ben Graham 代理 - 价值投资之父，只购买具有安全边际的隐藏瑰宝
3. Bill Ackman 代理 - 激进投资者，采取大胆立场并推动变革
4. Cathie Wood 代理 - 成长投资女王，坚信创新和颠覆的力量
5. Charlie Munger 代理 - 沃伦·巴菲特的合伙人，只以合理的价格购买优秀企业
6. Michael Burry 代理 - 擅长"大空头"式逆向投资，寻找深度价值
7. Peter Lynch 代理 - 务实投资者，在日常业务中寻找"十倍股"
8. Phil Fisher 代理 - 严谨的成长投资者，擅长深度"闲聊"式研究
9. Rakesh Jhunjhunwala 代理 - 印度大牛
10. Stanley Druckenmiller 代理 - 宏观传奇，寻找具有增长潜力的不对称机会
11. Warren Buffett 代理 - 奥马哈的先知，以合理价格寻找优秀公司
12. 估值代理 - 计算股票内在价值并生成交易信号
13. 情绪代理 - 分析市场情绪并生成交易信号
14. 基本面代理 - 分析基本面数据并生成交易信号
15. 技术分析代理 - 分析技术指标并生成交易信号
16. 风险管理器 - 计算风险指标并设置头寸限制
17. 投资组合经理 - 做出最终交易决策并生成订单

<img width="1042" alt="Screenshot 2025-03-22 at 6 19 07 PM" src="https://github.com/user-attachments/assets/cbae3dcf-b571-490d-b0ad-3f0f035ac0d4" />

**注意**：该系统模拟交易决策，并不实际进行交易。

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## 免责声明

本项目仅用于**教育和研究目的**。

- 不适用于实际交易或投资
- 不提供任何投资建议或保证
- 创建者不对任何财务损失承担责任
- 投资决策请咨询财务顾问
- 过往业绩不代表未来结果

使用本软件即表示您同意仅将其用于学习目的。

## 目录
- [设置](#设置)
  - [使用 uv](#使用-uv)
  - [使用 Docker](#使用-docker)
- [用法](#用法)
  - [运行对冲基金](#运行对冲基金)
  - [运行回测器](#运行回测器)
- [贡献](#贡献)
- [功能请求](#功能请求)
- [许可证](#许可证)

## 设置

### 使用 uv

克隆仓库：
```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

1. 安装 uv（如果尚未安装）：
```bash
# 在 macOS 和 Linux 上：
curl -LsSf https://astral.sh/uv/install.sh | sh

# 在 Windows 上：
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. 安装依赖：
```bash
uv sync
```

3. 设置您的环境变量：
```bash
# 为您的 API 密钥创建 .env 文件
cp .env.example .env
```

4. 设置您的 API 密钥：
```bash
# 用于运行 OpenAI 托管的 LLM（gpt-4o, gpt-4o-mini 等）
# 从 https://platform.openai.com/ 获取您的 OpenAI API 密钥
OPENAI_API_KEY=your-openai-api-key

# 用于运行 Groq 托管的 LLM（deepseek, llama3 等）
# 从 https://groq.com/ 获取您的 Groq API 密钥
GROQ_API_KEY=your-groq-api-key

# 用于获取对冲基金所需的金融数据
# 从 https://financialdatasets.ai/ 获取您的 Financial Datasets API 密钥
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
```

### 使用 Docker

1. 确保您的系统上已安装 Docker。如果未安装，可以从 [Docker 官方网站](https://www.docker.com/get-started) 下载。

2. 克隆仓库：
```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

3. 设置您的环境变量：
```bash
# 为您的 API 密钥创建 .env 文件
cp .env.example .env
```

4. 编辑 .env 文件以添加您的 API 密钥，如上所述。

5. 导航到 docker 目录：
```bash
cd docker
```

6. 构建 Docker 镜像：
```bash
# 在 Linux/Mac 上：
./run.sh build

# 在 Windows 上：
run.bat build
```

**重要**：您必须设置 `OPENAI_API_KEY`、`GROQ_API_KEY`、`ANTHROPIC_API_KEY` 或 `DEEPSEEK_API_KEY` 才能使对冲基金正常工作。如果您想使用所有提供商的 LLM，则需要设置所有 API 密钥。

AAPL、GOOGL、MSFT、NVDA 和 TSLA 的金融数据是免费的，不需要 API 密钥。

对于任何其他股票代码，您需要在 .env 文件中设置 `FINANCIAL_DATASETS_API_KEY`。

## 用法

### 运行对冲基金

#### 使用 uv
```bash
uv run python src/main.py --ticker AAPL,MSFT,NVDA
```

#### 使用 Docker
**注意**：所有 Docker 命令都必须在 `docker/` 目录下运行。

```bash
# 首先导航到 docker 目录
cd docker

# 在 Linux/Mac 上：
./run.sh --ticker AAPL,MSFT,NVDA main

# 在 Windows 上：
run.bat --ticker AAPL,MSFT,NVDA main
```

**示例输出：**
<img width="992" alt="Screenshot 2025-01-06 at 5 50 17 PM" src="https://github.com/user-attachments/assets/e8ca04bf-9989-4a7d-a8b4-34e04666663b" />

您还可以指定 `--ollama` 标志以使用本地 LLM 运行 AI 对冲基金。

```bash
# 使用 uv：
uv run python src/main.py --ticker AAPL,MSFT,NVDA --ollama

# 使用 Docker（从 docker/ 目录）：
# 在 Linux/Mac 上：
./run.sh --ticker AAPL,MSFT,NVDA --ollama main

# 在 Windows 上：
run.bat --ticker AAPL,MSFT,NVDA --ollama main
```

您还可以指定 `--show-reasoning` 标志以将每个代理的推理打印到控制台。

```bash
# 使用 uv：
uv run python src/main.py --ticker AAPL,MSFT,NVDA --show-reasoning

# 使用 Docker（从 docker/ 目录）：
# 在 Linux/Mac 上：
./run.sh --ticker AAPL,MSFT,NVDA --show-reasoning main

# 在 Windows 上：
run.bat --ticker AAPL,MSFT,NVDA --show-reasoning main
```

您可以选择指定开始和结束日期，以便在特定时间段内做出决策。

```bash
# 使用 uv：
uv run python src/main.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 

# 使用 Docker（从 docker/ 目录）：
# 在 Linux/Mac 上：
./run.sh --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 main

# 在 Windows 上：
run.bat --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 main
```

### 运行回测器

#### 使用 uv
```bash
uv run python src/backtester.py --ticker AAPL,MSFT,NVDA
```

#### 使用 Docker
**注意**：所有 Docker 命令都必须在 `docker/` 目录下运行。

```bash
# 首先导航到 docker 目录
cd docker

# 在 Linux/Mac 上：
./run.sh --ticker AAPL,MSFT,NVDA backtest

# 在 Windows 上：
run.bat --ticker AAPL,MSFT,NVDA backtest
```

**示例输出：**
<img width="941" alt="Screenshot 2025-01-06 at 5 47 52 PM" src="https://github.com/user-attachments/assets/00e794ea-8628-44e6-9a84-8f8a31ad3b47" />

您可以选择指定开始和结束日期以在特定时间段内进行回测。

```bash
# 使用 uv：
uv run python src/backtester.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01

# 使用 Docker（从 docker/ 目录）：
# 在 Linux/Mac 上：
./run.sh --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 backtest

# 在 Windows 上：
run.bat --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 backtest
```

您还可以指定 `--ollama` 标志以使用本地 LLM 运行回测器。
```bash
# 使用 uv：
uv run python src/backtester.py --ticker AAPL,MSFT,NVDA --ollama

# 使用 Docker（从 docker/ 目录）：
```

## 贡献

1. Fork 仓库
2. 创建功能分支
3. 提交您的更改
4. Push 到分支
5. 创建 Pull Request

**重要**：请保持您的 Pull Request 小而专注。这将使其更容易审查和合并。

## 功能请求

如果您有功能请求，请打开一个 [issue](https://github.com/virattt/ai-hedge-fund/issues) 并确保其标记为 `enhancement`。

## 许可证

本项目采用 MIT 许可证 - 有关详细信息，请参阅 LICENSE 文件。 