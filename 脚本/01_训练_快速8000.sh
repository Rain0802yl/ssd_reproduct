#!/bin/bash
# ============================================================================
# 训练 SSD300（VOC2007 trainval，8000 迭代快速验证）
#
# 预计耗时：约 3-4 小时（Apple M5 / MPS，batch=16）
# 产出目录：experiments/quick8000/
#   config.yaml        本次运行的完整配置
#   train_log.csv      训练日志（画曲线用）
#   last.pth           最新权重
#   iter_XXXX.pth      周期性快照
#
# 想要中断后继续训练，用 脚本/02_继续训练.sh
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/环境变量.sh" || exit 1

"$PYTHON" -m src.train \
    --config src/configs/ssd300_voc_quick.yaml \
    --exp-name quick8000
