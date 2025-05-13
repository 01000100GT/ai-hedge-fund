# AI 对冲基金

这是一个AI驱动的对冲基金概念验证项目。该项目的目标是探索使用人工智能进行交易决策。本项目**仅供教育**目的使用，不适用于实际交易或投资。

该系统由多个协同工作的智能体组成：

1. 本杰明·格雷厄姆智能体 - 价值投资之父，只买具有安全边际的隐藏宝石
2. 比尔·阿克曼智能体 - 激进投资者，采取大胆立场并推动变革
3. 凯茜·伍德智能体 - 成长型投资女王，相信创新和颠覆的力量
4. 查理·芒格智能体 - 沃伦·巴菲特的合伙人，以合理价格只买优质企业
5. 迈克尔·伯里智能体 - 《大空头》中的逆势者，寻找深度价值
6. 彼得·林奇智能体 - 务实的投资者，在日常企业中寻找"十倍股"
7. 菲利普·费舍尔智能体 - 精细的成长型投资者，使用深度"小道消息"研究
8. 斯坦利·德鲁肯米勒智能体 - 宏观传奇人物，寻找具有增长潜力的不对称机会
9. 沃伦·巴菲特智能体 - 奥马哈的先知，寻找以合理价格买入的优质公司
10. 估值智能体 - 计算股票的内在价值并生成交易信号
11. 情绪智能体 - 分析市场情绪并生成交易信号
12. 基本面智能体 - 分析基本面数据并生成交易信号
13. 技术面智能体 - 分析技术指标并生成交易信号
14. 风险管理者 - 计算风险指标并设置持仓限制
15. 投资组合管理者 - 做出最终交易决策并生成订单
    
<img width="1042" alt="Screenshot 2025-03-22 at 6 19 07 PM" src="https://github.com/user-attachments/assets/cbae3dcf-b571-490d-b0ad-3f0f035ac0d4" />


**注意**：系统模拟交易决策，不实际进行交易。

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## 免责声明

本项目**仅供教育和研究目的使用**。

- 不用于实际交易或投资
- 不提供任何保证或担保
- 过去的表现不代表未来的结果
- 创建者对财务损失不承担任何责任
- 投资决策请咨询财务顾问

使用本软件即表示您同意仅将其用于学习目的。

## 目录
- [设置](#设置)
  - [使用Poetry](#使用poetry)
  - [使用Docker](#使用docker)
- [使用方法](#使用方法)
  - [运行对冲基金](#运行对冲基金)
  - [运行回测器](#运行回测器)
- [项目结构](#项目结构)
- [贡献](#贡献)
- [功能请求](#功能请求)
- [许可证](#许可证)

## 设置

### 使用Poetry

克隆仓库：
```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

1. 安装Poetry（如果尚未安装）：
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. 安装依赖：
```bash
poetry install
```

3. 设置环境变量：
```bash
# 为API密钥创建.env文件
cp .env.example .env
```

4. 设置API密钥：
```bash
# 用于运行由OpenAI托管的LLM（gpt-4o, gpt-4o-mini等）
# 从https://platform.openai.com/获取OpenAI API密钥
OPENAI_API_KEY=your-openai-api-key

# 用于运行由Groq托管的LLM（deepseek, llama3等）
# 从https://groq.com/获取Groq API密钥
GROQ_API_KEY=your-groq-api-key

# 用于获取为对冲基金提供动力的金融数据
# 从https://financialdatasets.ai/获取Financial Datasets API密钥
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
```

### 使用Docker

1. 确保您的系统上已安装Docker。如果没有，可以从[Docker官方网站](https://www.docker.com/get-started)下载。

2. 克隆仓库：
```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

3. 设置环境变量：
```bash
# 为API密钥创建.env文件
cp .env.example .env
```

4. 编辑.env文件，按上述说明添加API密钥。

5. 构建Docker镜像：
```bash
# 在Linux/Mac上：
./run.sh build

# 在Windows上：
run.bat build
```

**重要提示**：对冲基金需要设置`OPENAI_API_KEY`、`GROQ_API_KEY`、`ANTHROPIC_API_KEY`或`DEEPSEEK_API_KEY`才能工作。如果要使用所有提供商的LLM，需要设置所有API密钥。

AAPL、GOOGL、MSFT、NVDA和TSLA的金融数据是免费的，不需要API密钥。

对于任何其他股票代码，需要在.env文件中设置`FINANCIAL_DATASETS_API_KEY`。

## 使用方法

### 运行对冲基金

#### 使用Poetry
```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA
```

#### 使用Docker
```bash
# 在Linux/Mac上：
./run.sh --ticker AAPL,MSFT,NVDA main

# 在Windows上：
run.bat --ticker AAPL,MSFT,NVDA main
```

**示例输出：**
<img width="992" alt="Screenshot 2025-01-06 at 5 50 17 PM" src="https://github.com/user-attachments/assets/e8ca04bf-9989-4a7d-a8b4-34e04666663b" />

您还可以指定`--ollama`标志，使用本地LLM运行AI对冲基金。

```bash
# 使用Poetry：
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --ollama

# 使用Docker（在Linux/Mac上）：
./run.sh --ticker AAPL,MSFT,NVDA --ollama main

# 使用Docker（在Windows上）：
run.bat --ticker AAPL,MSFT,NVDA --ollama main
```

您还可以指定`--show-reasoning`标志，将每个智能体的推理过程打印到控制台。

```bash
# 使用Poetry：
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --show-reasoning

# 使用Docker（在Linux/Mac上）：
./run.sh --ticker AAPL,MSFT,NVDA --show-reasoning main

# 使用Docker（在Windows上）：
run.bat --ticker AAPL,MSFT,NVDA --show-reasoning main
```

您可以选择指定开始和结束日期，为特定时间段做出决策。

```bash
# 使用Poetry：
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 

# 使用Docker（在Linux/Mac上）：
./run.sh --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 main

# 使用Docker（在Windows上）：
run.bat --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 main
```

### 运行回测器

#### 使用Poetry
```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA
```

#### 使用Docker
```bash
# 在Linux/Mac上：
./run.sh --ticker AAPL,MSFT,NVDA backtest

# 在Windows上：
run.bat --ticker AAPL,MSFT,NVDA backtest
```

**示例输出：**
<img width="941" alt="Screenshot 2025-01-06 at 5 47 52 PM" src="https://github.com/user-attachments/assets/00e794ea-8628-44e6-9a84-8f8a31ad3b47" />


您可以选择指定开始和结束日期，在特定时间段内进行回测。

```bash
# 使用Poetry：
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01

# 使用Docker（在Linux/Mac上）：
./run.sh --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 backtest

# 使用Docker（在Windows上）：
run.bat --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 backtest
```

您还可以指定`--ollama`标志，使用本地LLM运行回测器。
```bash
# 使用Poetry：
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA --ollama

# 使用Docker（在Linux/Mac上）：
./run.sh --ticker AAPL,MSFT,NVDA --ollama backtest

# 使用Docker（在Windows上）：
run.bat --ticker AAPL,MSFT,NVDA --ollama backtest
```


## 项目结构 
```
ai-hedge-fund/
├── src/
│   ├── agents/                   # 智能体定义和工作流程
│   │   ├── bill_ackman.py        # 比尔·阿克曼智能体
│   │   ├── fundamentals.py       # 基本面分析智能体
│   │   ├── portfolio_manager.py  # 投资组合管理智能体
│   │   ├── risk_manager.py       # 风险管理智能体
│   │   ├── sentiment.py          # 情绪分析智能体
│   │   ├── technicals.py         # 技术分析智能体
│   │   ├── valuation.py          # 估值分析智能体
│   │   ├── ...                   # 其他智能体
│   │   ├── warren_buffett.py     # 沃伦·巴菲特智能体
│   ├── tools/                    # 智能体工具
│   │   ├── api.py                # API工具
│   ├── backtester.py             # 回测工具
│   ├── main.py # 主入口点
├── pyproject.toml
├── ...
```

## 贡献

1. Fork仓库
2. 创建功能分支
3. 提交您的更改
4. 推送到分支
5. 创建Pull Request

**重要提示**：请保持您的pull request小而集中。这将使审核和合并更容易。

## 功能请求

如果您有功能请求，请开一个[issue](https://github.com/virattt/ai-hedge-fund/issues)，并确保标记为`enhancement`。

## 许可证

本项目根据MIT许可证授权 - 详情请参阅LICENSE文件。 