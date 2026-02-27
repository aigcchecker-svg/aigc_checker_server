#!/usr/bin/env bash
set -e

# 切换到脚本所在目录（serverapi/）
cd "$(dirname "$0")"

# 检查 Python 虚拟环境，不存在则自动创建
if [ ! -d ".venv" ]; then
    echo ">>> 未检测到虚拟环境，正在创建 .venv ..."
    python3 -m venv .venv
fi

# 激活虚拟环境
source .venv/bin/activate

# 安装/更新依赖
echo ">>> 安装依赖..."
pip install -q -r requirements.txt

# 加载 .env 文件（如果存在）
if [ -f ".env" ]; then
    echo ">>> 加载 .env 配置..."
    export $(grep -v '^#' .env | xargs)
fi

# 提示访问地址
echo ""
echo "=========================================="
echo "  AIGC Checker Engine 启动中..."
echo "  访问地址: http://127.0.0.1:8027"
echo "  API 文档: http://127.0.0.1:8027/docs"
echo "  按 Ctrl+C 停止服务"
echo "=========================================="
echo ""

# 启动服务（开发模式加 --reload，生产模式去掉）
uvicorn main:app --host 0.0.0.0 --port 8027 --reload
