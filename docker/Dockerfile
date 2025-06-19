# 基于Python 3.11精简版镜像
FROM python:3.11-slim

# 设置工作目录为/app
WORKDIR /app

# Set PYTHONPATH to include the app directory
ENV PYTHONPATH=/app

# Install Poetry
RUN pip install poetry==1.7.1

# Copy only dependency files first for better caching
COPY ../pyproject.toml ../poetry.lock* /app/

# 配置Poetry不使用虚拟环境并安装依赖
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy rest of the source code
COPY ../ /app/

# Default command (will be overridden by Docker Compose)
CMD ["python", "src/main.py"] 
