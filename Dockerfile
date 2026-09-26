# GraphForge 运行镜像
# 说明：基础镜像取 python:3.11-slim，依赖走 requirements.txt（核心块，非 lock），
#       避免 requirements.lock.txt 中个别版本在 3.11 上无 wheel。
FROM python:3.11-slim

LABEL maintainer="晨星" \
      description="GraphForge: graph embedding benchmark (node classification / link prediction)"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRAPHFORGE_RANDOM_STATE=42 \
    GRAPHFORGE_OUTPUT_DIR=/app/artifacts

WORKDIR /app

# 只安装核心依赖（可选依赖 karateclub / node2vec 在多数环境不可用，系统会自动降级）
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

# 健康检查：导入包 + 打印后端可用性（不抛异常）
RUN python -c "import graphforge; print(graphforge.__version__); print(graphforge.available_backends())"

CMD ["python", "-m", "graphforge.cli", "doctor"]
