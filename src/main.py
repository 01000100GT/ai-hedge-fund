# src/main.py
# 导入 sys 模块，用于与 Python 解释器交互，如此处用于退出程序。
import sys

# 导入 dotenv 模块的 load_dotenv 函数，用于从 .env 文件加载环境变量。
from dotenv import load_dotenv

# 从 langchain_core.messages 导入 HumanMessage 类，用于表示用户输入的消息。
from langchain_core.messages import HumanMessage

# 从 langgraph.graph 导入 END 和 StateGraph，用于构建和管理状态图（Agent 工作流）。
from langgraph.graph import END, StateGraph

# 从 colorama 导入 Fore, Style, init，用于在终端输出带颜色的文本。
from colorama import Fore, Style, init

# 导入 questionary 库，用于创建交互式命令行提示。
import questionary

# 从 src.agents.portfolio_manager 导入 portfolio_management_agent，这是投资组合管理 Agent 的处理函数。
from src.agents.portfolio_manager import portfolio_management_agent

# 从 src.agents.risk_manager 导入 risk_management_agent，这是风险管理 Agent 的处理函数。
from src.agents.risk_manager import risk_management_agent

# 从 src.graph.state 导入 AgentState 类，定义了 Agent 工作流的状态结构。
from src.graph.state import AgentState

# 从 src.utils.display 导入 print_trading_output 函数，用于格式化并打印最终的交易决策。
from src.utils.display import print_trading_output

# 从 src.utils.analysts 导入 ANALYST_ORDER 和 get_analyst_nodes，用于获取分析师的顺序和对应的节点信息。
from src.utils.analysts import ANALYST_ORDER, get_analyst_nodes

# 从 src.utils.progress 导入 progress 对象，用于显示长时间运行任务的进度指示。
from src.utils.progress import progress

# 从 src.llm.models 导入模型相关的常量和函数，如模型列表、获取模型信息、模型提供商枚举。
from src.llm.models import LLM_ORDER, OLLAMA_LLM_ORDER, get_model_info, ModelProvider

# 从 src.utils.ollama 导入 ensure_ollama_and_model 函数，用于检查 Ollama 服务和模型是否可用。
from src.utils.ollama import ensure_ollama_and_model

# 导入 argparse 模块，用于解析命令行参数。
import argparse

# 导入 datetime 类，用于处理日期和时间。
from datetime import datetime

# 从 dateutil.relativedelta 导入 relativedelta，用于方便地进行日期计算（如加减月份）。
from dateutil.relativedelta import relativedelta

# 从 src.utils.visualize 导入 save_graph_as_png 函数，用于将 Agent 工作流图保存为图片。
from src.utils.visualize import save_graph_as_png

# 导入 json 模块，用于处理 JSON 数据。
import json

# Load environment variables from .env file
# 执行 load_dotenv() 来加载 .env 文件中定义的环境变量。
load_dotenv()

# 使用 colorama 的 init 函数初始化，autoreset=True 确保颜色和样式在每次打印后自动重置。
init(autoreset=True)


# 定义函数 parse_hedge_fund_response，用于解析 LLM 返回的 JSON 格式的交易决策。
def parse_hedge_fund_response(response):
    """
    解析一个 JSON 字符串并返回一个 Python 字典。
    处理可能的 JSON 解码错误和类型错误。

    Args:
        response: 期望是包含 JSON 数据的字符串。

    Returns:
        dict: 解析后的 Python 字典。
        None: 如果解析失败或输入类型不正确。
    """
    # 尝试使用 json.loads() 解析输入的 response 字符串。
    try:
        return json.loads(response)
    # 捕获 JSON 解码错误。
    except json.JSONDecodeError as e:
        # 打印错误信息和导致错误的原始响应。
        print(f"JSON decoding error: {e}\nResponse: {repr(response)}")
        # 返回 None 表示解析失败。
        return None
    # 捕获类型错误（例如，如果传入的不是字符串）。
    except TypeError as e:
        # 打印类型错误信息。
        print(f"Invalid response type (expected string, got {type(response).__name__}): {e}")
        # 返回 None 表示输入类型不正确。
        return None
    # 捕获其他意外错误。
    except Exception as e:
        # 打印意外错误信息和原始响应。
        print(f"Unexpected error while parsing response: {e}\nResponse: {repr(response)}")
        # 返回 None 表示发生意外错误。
        return None


##### Run the Hedge Fund #####
# 定义函数 run_hedge_fund，核心逻辑，运行整个 AI 对冲基金模拟流程。
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4o",
    model_provider: str = "OpenAI",
):
    """
    运行 AI 对冲基金模拟。

    Args:
        tickers (list[str]): 需要分析的股票代码列表。
        start_date (str): 数据分析的开始日期 (YYYY-MM-DD)。
        end_date (str): 数据分析的结束日期 (YYYY-MM-DD)。
        portfolio (dict): 当前的投资组合状态（现金、持仓等）。
        show_reasoning (bool, optional): 是否显示每个 Agent 的决策推理过程。默认为 False。
        selected_analysts (list[str], optional): 用户选择的分析师 Agent 列表。默认为空列表（使用所有默认分析师）。
        model_name (str, optional): 使用的 LLM 模型名称。默认为 "gpt-4o"。
        model_provider (str, optional): 使用的 LLM 提供商 (如 "OpenAI", "Ollama")。默认为 "OpenAI"。

    Returns:
        dict: 包含最终交易决策 ('decisions') 和各分析师信号 ('analyst_signals') 的字典。
              如果解析失败，'decisions' 可能为 None。
    """
    # Start progress tracking
    # 启动进度条显示。
    progress.start()

    # 使用 try...finally 确保进度条总能被停止。
    try:
        # Create a new workflow if analysts are customized
        # 检查用户是否自定义了分析师列表。
        if selected_analysts:
            # 如果自定义了分析师，则动态创建包含这些分析师的工作流。
            workflow = create_workflow(selected_analysts)
            # 编译工作流，得到可执行的 agent。
            agent = workflow.compile()
        # 如果没有自定义分析师。
        else:
            # 使用默认的、预编译好的工作流 'app' (假设 'app' 在全局作用域已定义并编译好)。
            agent = app  # 注意：这里的 'app' 变量是在 __main__ 块中创建和编译的，如果直接调用此函数可能需要调整

        # 调用编译好的 agent (LangGraph 工作流) 并传入初始状态。
        final_state = agent.invoke(
            {
                # "messages" 包含初始指令。
                "messages": [
                    HumanMessage(
                        content="Make trading decisions based on the provided data.",
                    )
                ],
                # "data" 包含运行所需的业务数据。
                "data": {
                    "tickers": tickers,  # 股票列表
                    "portfolio": portfolio,  # 投资组合
                    "start_date": start_date,  # 开始日期
                    "end_date": end_date,  # 结束日期
                    "analyst_signals": {},  # 初始化用于存储分析师信号的字典
                },
                # "metadata" 包含运行的元数据/配置。
                "metadata": {
                    "show_reasoning": show_reasoning,  # 是否显示推理
                    "model_name": model_name,  # LLM 模型名
                    "model_provider": model_provider,  # LLM 提供商
                },
            },
        )

        # 返回包含最终决策和分析师信号的字典。
        return {
            # 解析最终状态中最后一条消息的内容（通常是 Portfolio Manager Agent 返回的 JSON 决策）。
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            # 获取最终状态中收集到的所有分析师信号。
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    # finally 块确保无论 try 中是否发生异常，都会执行。
    finally:
        # Stop progress tracking
        # 停止进度条显示。
        progress.stop()


# 定义函数 start，作为 LangGraph 工作流的起始节点。
def start(state: AgentState):
    """
    初始化工作流状态。这是工作流的入口点。

    Args:
        state (AgentState): 当前的工作流状态。

    Returns:
        AgentState: 原样返回状态，通常用于启动流程。
    """
    # 直接返回传入的状态，不进行修改。
    return state


# 定义函数 create_workflow，用于动态创建 LangGraph 工作流。
def create_workflow(selected_analysts=None):
    """
    根据选择的分析师动态创建 LangGraph 工作流。

    Args:
        selected_analysts (list[str], optional): 用户选择的分析师的键名列表。
                                                  如果为 None 或空列表，则使用所有默认分析师。

    Returns:
        StateGraph: 构建好的 LangGraph 工作流图。
    """
    # 创建一个 StateGraph 实例，状态类型为 AgentState。
    workflow = StateGraph(AgentState)
    # 添加一个名为 "start_node" 的起始节点，其处理函数为 start。
    workflow.add_node("start_node", start)

    # Get analyst nodes from the configuration
    # 从配置中获取所有可用的分析师节点及其处理函数。
    analyst_nodes = get_analyst_nodes()

    # Default to all analysts if none selected
    # 如果用户没有选择分析师 (selected_analysts 为 None 或空)，则默认使用所有可用的分析师。
    if selected_analysts is None:
        # 获取所有分析师节点的键名。
        selected_analysts = list(analyst_nodes.keys())
    # Add selected analyst nodes
    # 遍历用户选择的分析师键名。
    for analyst_key in selected_analysts:
        # 从 analyst_nodes 字典中获取对应的节点名称和处理函数。
        node_name, node_func = analyst_nodes[analyst_key]
        # 将选择的分析师节点添加到工作流中。
        workflow.add_node(node_name, node_func)
        # 添加入口点到所有分析师节点的边
        workflow.add_edge("start_node", node_name)

    # Always add risk and portfolio management
    # 总是添加风险管理 Agent 节点。
    workflow.add_node("risk_management_agent", risk_management_agent)
    # 总是添加投资组合管理 Agent 节点。
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)

    # Connect selected analysts to risk management
    # 将所有选择的分析师节点连接到风险管理节点。
    for analyst_key in selected_analysts:
        # 获取当前分析师的节点名称。
        node_name = analyst_nodes[analyst_key][0]
        # 添加从分析师节点到风险管理节点的边。
        workflow.add_edge(node_name, "risk_management_agent")

    # 添加从风险管理节点到投资组合管理节点的边。
    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    # 添加从投资组合管理节点到结束节点 (END) 的边。
    workflow.add_edge("portfolio_management_agent", END)

    # 设置工作流的入口点为 "start_node"。
    workflow.set_entry_point("start_node")
    # 返回构建好的工作流图。
    return workflow


# 程序的入口点，当脚本被直接运行时执行。
if __name__ == "__main__":
    # 创建 ArgumentParser 对象，用于解析命令行参数。
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
    # 添加 --initial-cash 参数，设置初始现金，默认为 100000.0。
    parser.add_argument("--initial-cash", type=float, default=100000.0, help="Initial cash position. Defaults to 100000.0)")
    # 添加 --margin-requirement 参数，设置初始保证金要求，默认为 0.0。
    parser.add_argument("--margin-requirement", type=float, default=0.0, help="Initial margin requirement. Defaults to 0.0")
    # 修改: 移除 required=True，添加从 README.md 获取的免费股票代码作为默认值。
    parser.add_argument(
        "--tickers",
        type=str,
        default="AAPL,GOOGL,MSFT,NVDA,TSLA", # 添加默认值
        help="Comma-separated list of stock ticker symbols. Defaults to AAPL,GOOGL,MSFT,NVDA,TSLA", # 更新帮助文本
    )
    # 添加 --start-date 参数，设置开始日期 (YYYY-MM-DD)，可选，默认为结束日期前3个月。
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Defaults to 3 months before end date",
    )
    # 添加 --end-date 参数，设置结束日期 (YYYY-MM-DD)，可选，默认为今天。
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD). Defaults to today")
    # 添加 --show-reasoning 参数，布尔标志，用于控制是否显示 Agent 推理过程。
    parser.add_argument("--show-reasoning", action="store_true", help="Show reasoning from each agent")
    # 添加 --show-agent-graph 参数，布尔标志，用于控制是否显示 Agent 图。
    parser.add_argument("--show-agent-graph", action="store_true", help="Show the agent graph")
    # 添加 --ollama 参数，布尔标志，用于控制是否使用本地 Ollama 进行 LLM 推理。
    parser.add_argument("--ollama", action="store_true", help="Use Ollama for local LLM inference")

    # 解析命令行传入的参数。
    args = parser.parse_args()

    # Parse tickers from comma-separated string
    # 将逗号分隔的股票代码字符串解析为列表，并去除两端空格。
    # 注意：现在即使不提供 --tickers 参数，args.tickers 也会有默认值 "AAPL,GOOGL,MSFT,NVDA,TSLA"
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]

    # Select analysts
    # 初始化选择的分析师列表为 None。
    selected_analysts = None
    # 使用 questionary 显示复选框，让用户选择要使用的 AI 分析师。
    choices = questionary.checkbox(
        "Select your AI analysts.",  # 提示信息
        # 选项列表，从 ANALYST_ORDER 构建，包含显示名称和值。
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        # 操作说明。
        instruction="\n\nInstructions: \n1. Press Space to select/unselect analysts.\n2. Press 'a' to select/unselect all.\n3. Press Enter when done to run the hedge fund.\n",
        # 验证规则：必须至少选择一个分析师。
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        # 设置交互界面的样式。
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
        # 执行提问并等待用户选择。
    ).ask()

    # 检查用户是否没有做出选择（可能按了 Ctrl+C）。
    if not choices:
        # 打印退出信息。
        print("\n\nInterrupt received. Exiting...")
        # 退出程序。
        sys.exit(0)
    # 如果用户做出了选择。
    else:
        # 将用户的选择赋值给 selected_analysts。
        selected_analysts = choices
        # 打印用户选择的分析师列表，使用绿色高亮显示。
        print(f"\nSelected analysts: {', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}\n")

    # Select LLM model based on whether Ollama is being used
    # 初始化模型选择和模型提供商为 None。
    model_choice = None
    model_provider = None

    # 检查命令行参数是否包含 --ollama。
    if args.ollama:
        # 打印提示信息，表明正在使用 Ollama。
        print(f"{Fore.CYAN}Using Ollama for local LLM inference.{Style.RESET_ALL}")

        # Select from Ollama-specific models
        # 使用 questionary 显示选择列表，让用户选择 Ollama 模型。
        model_choice = questionary.select(
            "Select your Ollama model:",  # 提示信息
            # 选项列表，从 OLLAMA_LLM_ORDER 构建。
            choices=[questionary.Choice(display, value=value) for display, value, _ in OLLAMA_LLM_ORDER],
            # 设置交互界面的样式。
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),
                    ("pointer", "fg:green bold"),
                    ("highlighted", "fg:green"),
                    ("answer", "fg:green bold"),
                ]
            ),
            # 执行提问并等待用户选择。
        ).ask()

        # 检查用户是否没有做出选择。
        if not model_choice:
            # 打印退出信息。
            print("\n\nInterrupt received. Exiting...")
            # 退出程序。
            sys.exit(0)

        # Ensure Ollama is installed, running, and the model is available
        # 调用 ensure_ollama_and_model 检查 Ollama 服务状态和所选模型是否可用。
        if not ensure_ollama_and_model(model_choice):
            # 如果检查失败，打印错误信息。
            print(f"{Fore.RED}Cannot proceed without Ollama and the selected model.{Style.RESET_ALL}")
            # 退出程序，返回错误码 1。
            sys.exit(1)

        # 设置模型提供商为 Ollama。
        model_provider = ModelProvider.OLLAMA.value
        # 打印用户选择的 Ollama 模型。
        print(f"\nSelected {Fore.CYAN}Ollama{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
    # 如果未使用 --ollama 参数。
    else:
        # Use the standard cloud-based LLM selection
        # 修改：在云模型选择列表中添加 OpenAI 兼容选项
        # 注意：这需要 src/llm/models.py 中的 LLM_ORDER 定义允许添加自定义选项，
        # 或者我们在这里手动构建 choices 列表。为简单起见，假设 LLM_ORDER 可以接受额外选项。
        # 这里我们创建一个新的 choices 列表来演示。
        cloud_model_choices = [questionary.Choice(display, value=value) for display, value, _ in LLM_ORDER]
        # 添加自定义 OpenAI 兼容选项
        cloud_model_choices.append(
            questionary.Choice(
                title="OpenAI Compatible (Custom Endpoint via env vars)", # 显示名称
                value="openai_compatible_custom" # 特殊标识符
            )
        )

        # 使用 questionary 显示选择列表，让用户选择云端 LLM 模型或自定义兼容端点。
        model_choice = questionary.select(
            "Select your LLM provider/model:",  # 更新提示信息
            choices=cloud_model_choices, # 使用包含新选项的列表
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),
                    ("pointer", "fg:green bold"),
                    ("highlighted", "fg:green"),
                    ("answer", "fg:green bold"),
                ]
            ),
            # 执行提问并等待用户选择。
        ).ask()

        # 检查用户是否没有做出选择。
        if not model_choice:
            # 打印退出信息。
            print("\n\nInterrupt received. Exiting...")
            # 退出程序。
            sys.exit(0)
        # 修改：添加处理自定义 OpenAI 兼容选项的逻辑
        elif model_choice == "openai_compatible_custom":
            # 设置特殊的模型提供商标示符
            # 注意：这需要在 src/llm/models.py 中定义对应的 ModelProvider 枚举值，例如 ModelProvider.OPENAI_COMPATIBLE
            model_provider = "OpenAICompatible" # 假设的提供者名称
            # 打印提示信息，告知用户需要设置环境变量
            print(f"\nSelected {Fore.CYAN}OpenAI Compatible (Custom Endpoint){Style.RESET_ALL}.")
            print(f"{Fore.YELLOW}Please ensure the following environment variables are set:")
            print(f"  - {Style.BRIGHT}OPENAI_API_BASE{Style.RESET_ALL}: The base URL of your OpenAI compatible endpoint.")
            print(f"  - {Style.BRIGHT}OPENAI_API_KEY{Style.RESET_ALL}: Your API key (if required by the endpoint).{Style.RESET_ALL}\n")
            # 使用一个通用的名称或者让用户在环境变量中指定模型
            model_name = "custom_openai_compatible_model" # 或者从环境变量读取
        # 如果用户选择了标准云模型。
        else:
            # Get model info using the helper function
            # 使用 get_model_info 函数获取所选模型的详细信息（包括提供商）。
            # 注意：get_model_info 函数需要能处理 LLM_ORDER 中定义的标准模型
            model_info = get_model_info(model_choice)
            # 检查是否成功获取到模型信息。
            if model_info:
                # 设置模型提供商。
                model_provider = model_info.provider.value
                # 打印用户选择的模型及其提供商。
                print(f"\nSelected {Fore.CYAN}{model_provider}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")
            # 如果未能获取模型信息。
            else:
                # 将模型提供商设为 "Unknown"。
                model_provider = "Unknown"
                # 仅打印用户选择的模型名称。
                print(f"\nSelected model: {Fore.GREEN + Style.BRIGHT}{model_choice}{Style.RESET_ALL}\n")

    # Create the workflow with selected analysts
    # 根据用户选择的分析师创建工作流。
    workflow = create_workflow(selected_analysts)
    # 编译工作流，得到可执行的 app 对象。
    app = workflow.compile()

    # 检查是否设置了 --show-agent-graph 标志。
    if args.show_agent_graph:
        # 初始化文件路径字符串。
        file_path = ""
        # 检查用户是否选择了分析师。
        if selected_analysts is not None:
            # 遍历选择的分析师，构建文件名。
            for selected_analyst in selected_analysts:
                file_path += selected_analyst + "_"
            # 添加文件后缀。
            file_path += "graph.png"
        # 调用函数将编译好的工作流图保存为 PNG 文件。
        save_graph_as_png(app, file_path)

    # Validate dates if provided
    # 检查用户是否提供了开始日期。
    if args.start_date:
        # 尝试将开始日期字符串解析为日期对象，以验证格式。
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
        # 如果格式不正确，捕获 ValueError。
        except ValueError:
            # 抛出更明确的错误信息。
            raise ValueError("Start date must be in YYYY-MM-DD format")

    # 检查用户是否提供了结束日期。
    if args.end_date:
        # 尝试将结束日期字符串解析为日期对象，以验证格式。
        try:
            datetime.strptime(args.end_date, "%Y-%m-%d")
        # 如果格式不正确，捕获 ValueError。
        except ValueError:
            # 抛出更明确的错误信息。
            raise ValueError("End date must be in YYYY-MM-DD format")

    # Set the start and end dates
    # 设置结束日期：如果用户提供了 args.end_date，则使用它；否则，使用当前日期。
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    # 设置开始日期：如果用户没有提供 args.start_date。
    if not args.start_date:
        # Calculate 3 months before end_date
        # 将结束日期字符串转换为日期对象。
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        # 计算结束日期前 3 个月的日期。
        start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
    # 如果用户提供了开始日期。
    else:
        # 直接使用用户提供的开始日期。
        start_date = args.start_date

    # Initialize portfolio with cash amount and stock positions
    # 初始化投资组合字典。
    portfolio = {
        # 设置初始现金。
        "cash": args.initial_cash,  # Initial cash amount
        # 设置初始保证金要求。
        "margin_requirement": args.margin_requirement,  # Initial margin requirement
        # 初始化已用保证金为 0.0。
        "margin_used": 0.0,  # total margin usage across all short positions
        # 初始化持仓信息字典。
        "positions": {
            # 遍历所有需要交易的股票代码。
            ticker: {
                "long": 0,  # 多头持股数量，初始化为 0。
                "short": 0,  # 空头持股数量，初始化为 0。
                "long_cost_basis": 0.0,  # 多头平均成本基础，初始化为 0.0。
                "short_cost_basis": 0.0,  # 空头平均卖出价格，初始化为 0.0。
                "short_margin_used": 0.0,  # 当前股票的空头已用保证金，初始化为 0.0。
            }
            # 列表推导，为每个 ticker 创建一个内部字典。
            for ticker in tickers
        },
        # 初始化已实现收益字典。
        "realized_gains": {
            # 遍历所有需要交易的股票代码。
            ticker: {
                "long": 0.0,  # 多头已实现收益，初始化为 0.0。
                "short": 0.0,  # 空头已实现收益，初始化为 0.0。
            }
            # 列表推导，为每个 ticker 创建一个内部字典。
            for ticker in tickers
        },
    }

    # Run the hedge fund
    # 调用 run_hedge_fund 函数执行主要的模拟流程。
    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        selected_analysts=selected_analysts,
        model_name=model_name, # 使用获取到的 model_name
        model_provider=model_provider, # 使用获取到的 model_provider
    )
    # 调用 print_trading_output 函数打印最终的交易结果。
    print_trading_output(result)
