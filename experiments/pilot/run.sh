#!/bin/bash
# Pilot 实验运行脚本

echo "========================================"
echo "DSPy 对抗优化 Pilot 实验"
echo "========================================"

# 检查 API Key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo ""
    echo "❌ Error: OPENROUTER_API_KEY 未设置"
    echo ""
    echo "请先设置 API Key:"
    echo "  export OPENROUTER_API_KEY='sk-or-v1-your-key-here'"
    echo ""
    exit 1
fi

# 检查依赖
echo ""
echo "检查依赖..."
pip install dspy-ai tqdm --break-system-packages -q 2>/dev/null

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 运行实验
echo ""
echo "开始运行实验..."
echo ""
python run_pilot.py

echo ""
echo "实验完成!"
