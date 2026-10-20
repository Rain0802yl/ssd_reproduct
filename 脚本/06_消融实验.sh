#!/bin/bash
# ============================================================================
# 【可选，目前暂缓】依次运行 4 组消融实验（A1~A4）
#
#   ⚠️ 这是主线跑完之后再考虑的事，用来做「组件贡献归因」，不影响主结果。
#      总耗时：约 18 小时（4 组 × 约 4.5 小时，均为 8000 迭代）
#
#   如果嫌久，可以给每组单独设置更小的预算（比如 4000 迭代，总共约 10 小时）：
#     在 src/configs/ablation_*.yaml 里加两行：
#         solver:
#           max_iter: 4000
#     注意：4 组必须用**同一个**迭代数才可比。
#
# 用法：
#   脚本/06_消融实验.sh            # 依次跑全部 4 组
#   脚本/06_消融实验.sh 1 3        # 只跑 A1 和 A3
#
# 每组跑完后会自动做一次评测并存图。
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/环境变量.sh" || exit 1

CONFIGS=(
  "ablation_A1_无数据增强"
  "ablation_A2_无多形状先验框"
  "ablation_A3_仅两层预测"
  "ablation_A4_关闭难负样本挖掘"
)

# 解析要跑哪几组
if [ $# -ge 2 ]; then
    SELECT=("$@")
else
    SELECT=(1 2 3 4)
fi

for i in "${SELECT[@]}"; do
    NAME="${CONFIGS[$((i-1))]}"
    if [ -z "$NAME" ]; then
        echo "无效的编号: $i（应为 1~4）"
        continue
    fi

    echo ""
    echo "############################################################"
    echo "#  消融实验 $i/4 : $NAME"
    echo "#  配置: src/configs/$NAME.yaml"
    echo "############################################################"

    "$PYTHON" -m src.train --config "src/configs/$NAME.yaml"
    if [ $? -ne 0 ]; then
        echo "训练失败，跳过该组的评测"
        continue
    fi

    EXP=$(grep -E '^\s+exp_name:' "src/configs/$NAME.yaml" | head -1 | sed 's/.*exp_name:[[:space:]]*//; s/"//g')
    echo "--- 评测 $EXP ---"
    "$PYTHON" -m src.eval \
        --config "src/configs/$NAME.yaml" \
        --weights "experiments/$EXP/last.pth" \
        --out "experiments/$EXP/eval"

    echo "--- 生成图表 $EXP ---"
    "$PYTHON" -m src.make_figures --exp-name "$EXP" --skip-detections
done

echo ""
echo "全部指定消融实验完成。结果在各 experiments/<实验名>/ 目录下。"
