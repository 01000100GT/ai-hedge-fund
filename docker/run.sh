#!/bin/bash

# 当提供 --help 参数时显示的帮助文本
show_help() {
  echo "AI 对冲基金 Docker 运行器"
  echo ""
  echo "用法: ./run.sh [选项] 命令"
  echo ""
  echo "选项:"
  echo "  --ticker SYMBOLS    股票代码列表，用逗号分隔 (例如: AAPL,MSFT,NVDA)"
  echo "  --start-date DATE   开始日期，格式为 YYYY-MM-DD"
  echo "  --end-date DATE     结束日期，格式为 YYYY-MM-DD"
  echo "  --initial-cash AMT  初始现金金额 (默认: 100000.0)"
  echo "  --margin-requirement RATIO  保证金要求比率 (默认: 0.0)"
  echo "  --ollama            使用 Ollama 进行本地 LLM 推理"
  echo "  --show-reasoning    显示每个代理的推理过程"
  echo ""
  echo "命令:"
  echo "  main                运行主要对冲基金应用"
  echo "  backtest           运行回测程序"
  echo "  build              构建 Docker 镜像"
  echo "  compose            使用 Docker Compose 运行（集成 Ollama）"
  echo "  ollama             仅启动 Ollama 容器用于模型管理"
  echo "  pull MODEL         将特定模型下载到 Ollama 容器中"
  echo "  help               显示此帮助信息"
  echo ""
  echo "示例:"
  echo "  ./run.sh --ticker AAPL,MSFT,NVDA main"
  echo "  ./run.sh --ticker AAPL,MSFT,NVDA --ollama main"
  echo "  ./run.sh --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 backtest"
  echo "  ./run.sh compose    # 使用 Docker Compose 运行（包含 Ollama）"
  echo "  ./run.sh ollama     # 仅启动 Ollama 容器"
  echo "  ./run.sh pull llama3 # 将 llama3 模型下载到 Ollama"
  echo ""
}

# 默认值
TICKER="AAPL,MSFT,NVDA"
USE_OLLAMA=""
START_DATE=""
END_DATE=""
INITIAL_AMOUNT="100000.0"
MARGIN_REQUIREMENT="0.0"
SHOW_REASONING=""
COMMAND=""
MODEL_NAME=""

# 解析参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --ticker)
      TICKER="$2"
      shift 2
      ;;
    --start-date)
      START_DATE="--start-date $2"
      shift 2
      ;;
    --end-date)
      END_DATE="--end-date $2"
      shift 2
      ;;
    --initial-cash)
      INITIAL_AMOUNT="$2"
      shift 2
      ;;
    --margin-requirement)
      MARGIN_REQUIREMENT="$2"
      shift 2
      ;;
    --ollama)
      USE_OLLAMA="--ollama"
      shift
      ;;
    --show-reasoning)
      SHOW_REASONING="--show-reasoning"
      shift
      ;;
    main|backtest|build|help|compose|ollama)
      COMMAND="$1"
      shift
      ;;
    pull)
      COMMAND="pull"
      MODEL_NAME="$2"
      shift 2
      ;;
    --help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# 检查是否提供了命令
if [ -z "$COMMAND" ]; then
  echo "错误：未指定命令。"
  show_help
  exit 1
fi

# 如果提供了 'help' 命令则显示帮助
if [ "$COMMAND" = "help" ]; then
  show_help
  exit 0
fi

# 检查 Docker Compose 是否存在
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
  echo "错误：未安装 Docker Compose。"
  exit 1
fi

# 确定使用哪个 Docker Compose 命令
if command -v docker-compose &> /dev/null; then
  COMPOSE_CMD="docker-compose"
else
  COMPOSE_CMD="docker compose"
fi

# 检测系统架构以配置 GPU
ARCH=$(uname -m)
OS=$(uname -s)
GPU_CONFIG=""

# 根据架构设置适当的 GPU 配置
if [ "$OS" = "Darwin" ] && { [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; }; then
  echo "检测到 Apple Silicon (M系列) - Metal GPU 加速应该已启用"
  # Metal GPU 通过 docker-compose.yml 中的环境变量处理
elif command -v nvidia-smi &> /dev/null; then
  echo "检测到 NVIDIA GPU - 添加 NVIDIA GPU 配置"
  GPU_CONFIG="-f docker-compose.yml -f docker-compose.nvidia.yml"
fi

# 如果提供了 'build' 命令则构建 Docker 镜像
if [ "$COMMAND" = "build" ]; then
  docker build -t ai-hedge-fund -f Dockerfile ..
  exit 0
fi

# 如果提供了 'ollama' 命令则启动 Ollama 容器
if [ "$COMMAND" = "ollama" ]; then
  echo "正在启动 Ollama 容器..."
  $COMPOSE_CMD $GPU_CONFIG up -d ollama
  
  # 检查 Ollama 是否正在运行
  echo "等待 Ollama 启动..."
  for i in {1..30}; do
    if docker run --rm --network=host curlimages/curl:latest curl -s http://localhost:11434/api/version &> /dev/null; then
      echo "Ollama 已经启动。"
      # 显示可用模型
      echo "可用模型："
      docker exec -t ollama ollama list
      
      echo -e "\n使用以下命令管理模型："
      echo "  ./run.sh pull <模型名称>   # 下载模型"
      echo "  ./run.sh ollama            # 启动 Ollama 并显示模型"
      exit 0
    fi
    echo -n "."
    sleep 1
  done
  
  echo "在预期时间内未能启动 Ollama。请检查容器日志。"
  exit 1
fi

# 如果提供了 'pull' 命令则拉取模型
if [ "$COMMAND" = "pull" ]; then
  if [ -z "$MODEL_NAME" ]; then
    echo "错误：未指定模型名称。"
    echo "用法: ./run.sh pull <模型名称>"
    echo "示例: ./run.sh pull llama3"
    exit 1
  fi
  
  # 如果 Ollama 未运行则启动它
  $COMPOSE_CMD $GPU_CONFIG up -d ollama
  
  # 等待 Ollama 启动
  echo "确保 Ollama 正在运行..."
  for i in {1..30}; do
    if docker run --rm --network=host curlimages/curl:latest curl -s http://localhost:11434/api/version &> /dev/null; then
      echo "Ollama 正在运行。"
      break
    fi
    echo -n "."
    sleep 1
  done
  
  # 拉取模型
  echo "正在拉取模型: $MODEL_NAME"
  echo "这可能需要一些时间，取决于模型大小和您的网络连接。"
  echo "您可以随时按 Ctrl+C 取消（模型将在后台继续下载）。"
  
  docker exec -t ollama ollama pull "$MODEL_NAME"
  
  # 检查模型是否成功拉取
  if docker exec -t ollama ollama list | grep -q "$MODEL_NAME"; then
    echo "模型 $MODEL_NAME 已成功下载。"
  else
    echo "警告：模型 $MODEL_NAME 可能未正确下载。"
    echo "使用以下命令检查 Ollama 容器状态: ./run.sh ollama"
  fi
  
  exit 0
fi

# 使用 Docker Compose 运行
if [ "$COMMAND" = "compose" ]; then
  echo "正在使用 Docker Compose 运行（包含 Ollama）..."
  $COMPOSE_CMD $GPU_CONFIG up --build
  exit 0
fi

# 检查 .env 文件是否存在，如果不存在则从 .env.example 创建
if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    echo "未找到 .env 文件。正在从 .env.example 创建..."
    cp .env.example .env
    echo "请编辑 .env 文件以添加您的 API 密钥。"
  else
    echo "错误：未找到 .env 或 .env.example 文件。"
    exit 1
  fi
fi

# 根据命令设置脚本路径和参数
if [ "$COMMAND" = "main" ]; then
  SCRIPT_PATH="src/main.py"
  if [ "$COMMAND" = "main" ]; then
    INITIAL_PARAM="--initial-cash $INITIAL_AMOUNT"
  fi
elif [ "$COMMAND" = "backtest" ]; then
  SCRIPT_PATH="src/backtester.py"
  if [ "$COMMAND" = "backtest" ]; then
    INITIAL_PARAM="--initial-capital $INITIAL_AMOUNT"
  fi
fi

# 如果使用 Ollama，确保服务已启动
if [ -n "$USE_OLLAMA" ]; then
  echo "正在设置 Ollama 容器用于本地 LLM 推理..."
  
  # 如果 Ollama 容器未运行则启动它
  $COMPOSE_CMD $GPU_CONFIG up -d ollama
  
  # 等待 Ollama 启动
  echo "等待 Ollama 启动..."
  for i in {1..30}; do
    if docker run --rm --network=host curlimages/curl:latest curl -s http://localhost:11434/api/version &> /dev/null; then
      echo "Ollama 正在运行。"
      # 显示可用模型
      echo "可用模型："
      docker exec -t ollama ollama list
      break
    fi
    echo -n "."
    sleep 1
  done
  
  # 如果需要则构建 AI 对冲基金镜像
  if [[ "$(docker images -q ai-hedge-fund 2> /dev/null)" == "" ]]; then
    echo "正在构建 AI 对冲基金镜像..."
    docker build -t ai-hedge-fund .
  fi
  
  # 创建 Docker Compose 命令覆盖
  COMMAND_OVERRIDE=""
  
  if [ -n "$START_DATE" ]; then
    COMMAND_OVERRIDE="$COMMAND_OVERRIDE $START_DATE"
  fi
  
  if [ -n "$END_DATE" ]; then
    COMMAND_OVERRIDE="$COMMAND_OVERRIDE $END_DATE"
  fi
  
  if [ -n "$INITIAL_PARAM" ]; then
    COMMAND_OVERRIDE="$COMMAND_OVERRIDE $INITIAL_PARAM"
  fi
  
  if [ -n "$MARGIN_REQUIREMENT" ]; then
    COMMAND_OVERRIDE="$COMMAND_OVERRIDE --margin-requirement $MARGIN_REQUIREMENT"
  fi
  
  # 使用 Docker Compose 运行命令
  echo "正在使用 Docker Compose 运行带 Ollama 的 AI 对冲基金..."
  
  # 根据命令和推理标志使用适当的服务
  if [ "$COMMAND" = "main" ]; then
    if [ -n "$SHOW_REASONING" ]; then
      $COMPOSE_CMD $GPU_CONFIG run --rm hedge-fund-reasoning python src/main.py --ticker $TICKER $COMMAND_OVERRIDE $SHOW_REASONING --ollama
    else
      $COMPOSE_CMD $GPU_CONFIG run --rm hedge-fund-ollama python src/main.py --ticker $TICKER $COMMAND_OVERRIDE --ollama
    fi
  elif [ "$COMMAND" = "backtest" ]; then
    $COMPOSE_CMD $GPU_CONFIG run --rm backtester-ollama python src/backtester.py --ticker $TICKER $COMMAND_OVERRIDE $SHOW_REASONING --ollama
  fi
  
  exit 0
fi

# 标准 Docker 运行（不使用 Ollama）
# 构建命令
CMD="docker run -it --rm -v $(pwd)/.env:/app/.env"

# Add the command
CMD="$CMD ai-hedge-fund python $SCRIPT_PATH --ticker $TICKER $START_DATE $END_DATE $INITIAL_PARAM --margin-requirement $MARGIN_REQUIREMENT $SHOW_REASONING"

# 运行命令
echo "正在运行: $CMD"
$CMD 