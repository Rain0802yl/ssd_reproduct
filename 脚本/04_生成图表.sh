#!/bin/bash
# ============================================================================
# 生成报告用的全部图表
# 用法：脚本/04_生成图表.sh [实验名]
#
# 产出：experiments/<实验名>/figures/
#   training_curves.png   训练损失与学习率曲线
#   ap_bar.png            20 类 AP 柱状图
#   pr_curves.png         各类 PR 曲线
#   detections/           检测结果可视化（GT vs 预测）
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/环境变量.sh" || exit 1

EXP="${1:-quick8000}"
"$PYTHON" -m src.make_figures --exp-name "$EXP"
