# 基于Python 3.11精简版镜像
FROM python:3.11-slim

# 设置工作目录为/app
WORKDIR /app

# 安装Poetry包管理工具
RUN pip install poetry==1.7.1

# 首先只复制依赖文件以便更好地利用缓存
COPY pyproject.toml poetry.lock* /app/

# 配置Poetry不使用虚拟环境并安装依赖
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# 复制其余源代码
COPY . /app/

# 默认命令（将被Docker Compose覆盖）
CMD ["python", "src/main.py"]