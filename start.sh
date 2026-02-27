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

# 询问是否为生产环境
read -r -p ">>> 是否为线上生产环境？[y/N] " IS_PROD

echo ""
echo "=========================================="
echo "  AIGC Checker Engine 启动中..."
echo "  访问地址: http://127.0.0.1:8027"
echo "  API 文档: http://127.0.0.1:8027/docs"

if [[ "$IS_PROD" =~ ^[Yy]$ ]]; then
    echo "  模式: 生产（后台常驻进程）"
    echo "  日志: nohup.out"
    echo "  停止: kill \$(cat server.pid)"
else
    echo "  模式: 开发（--reload，按 Ctrl+C 停止）"
fi

echo "=========================================="
echo ""

# 启动服务
if [[ "$IS_PROD" =~ ^[Yy]$ ]]; then
    nohup uvicorn main:app --host 0.0.0.0 --port 8027 > nohup.out 2>&1 &
    echo $! > server.pid
    echo ">>> 服务已在后台启动，PID: $(cat server.pid)"
else
    uvicorn main:app --host 0.0.0.0 --port 8027 --reload
fi
