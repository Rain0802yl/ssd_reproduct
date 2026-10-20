#!/bin/bash
# ============================================================================
# 在 VOC2007 test（4952 张）上评测，计算 mAP
# 用法：脚本/03_评测_VOC2007test.sh [实验名] [权重路径]
#
# 产出：experiments/<实验名>/eval/
#   metrics.json    mAP 与 20 类逐类 AP
#   pr_curves.npz   PR 曲线数据
# 终端会直接打印一张 AP 表，可以直接截图放进报告。
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/环境变量.sh" || exit 1

EXP="${1:-quick8000}"
W="${2:-experiments/$EXP/last.pth}"

if [ ! -f "$W" ]; then
    echo "找不到权重文件: $W"
    exit 1
fi

"$PYTHON" -m src.eval \
    --config src/configs/ssd300_voc_quick.yaml \
    --weights "$W" \
    --out "experiments/$EXP/eval"
