#!/bin/bash
# ============================================================================
# 从快照继续训练（断点续训）
#
# 用法：
#   脚本/02_继续训练.sh                              # 续训 quick8000
#   脚本/02_继续训练.sh long15000                    # 续训 long15000（自动选对配置）
#   脚本/02_继续训练.sh long15000 <快照路径>          # 指定快照
#   脚本/02_继续训练.sh my_exp <快照> <配置文件>      # 完整指定
#
# 配置文件的自动选择规则：
#   实验名里含 "long" -> ssd300_voc_long.yaml（15000 迭代）
#   其他              -> ssd300_voc_quick.yaml（8000 迭代）
#   ⚠️ 一定要用与当初训练时相同的配置，否则 max_iter / 里程碑会错乱。
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/环境变量.sh" || exit 1

EXP="${1:-quick8000}"
SNAP="${2:-experiments/$EXP/last.pth}"
CONFIG="${3:-}"

# 没显式指定配置时，按实验名推断
if [ -z "$CONFIG" ]; then
    case "$EXP" in
        *long*) CONFIG="src/configs/ssd300_voc_long.yaml" ;;
        *)      CONFIG="src/configs/ssd300_voc_quick.yaml" ;;
    esac
fi

if [ ! -f "$SNAP" ]; then
    echo "❌ 找不到快照文件: $SNAP"
    echo "   可用的快照："
    ls -1 experiments/*/*.pth 2>/dev/null | sed 's/^/     /' || echo "     (没有任何快照)"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "❌ 找不到配置文件: $CONFIG"
    exit 1
fi

echo "============================================================"
echo "  续训实验名 : $EXP"
echo "  快照文件   : $SNAP"
echo "  使用配置   : $CONFIG"
echo "============================================================"

"$PYTHON" -m src.train \
    --config "$CONFIG" \
    --exp-name "$EXP" \
    --resume "$SNAP"
