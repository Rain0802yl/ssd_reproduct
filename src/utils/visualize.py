"""可视化工具：检测结果、训练曲线、PR 曲线、AP 柱状图。

产出的图可直接插入报告。
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# 每个类别的固定颜色（BGR），保证多次画图配色一致
_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


def color_for(label: int) -> Tuple[int, int, int]:
    return _PALETTE[label % len(_PALETTE)]


def draw_boxes(
    img_bgr: np.ndarray,
    boxes: np.ndarray,
    labels: Sequence[int],
    scores: Optional[Sequence[float]] = None,
    class_names: Sequence[str] = (),
    thickness: int = 2,
    font_scale: float = 0.5,
    ground_truth: bool = False,
) -> np.ndarray:
    """在图像上画框。boxes 为归一化角点坐标 (N,4)。"""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    for i, box in enumerate(boxes):
        x1 = int(round(box[0] * w))
        y1 = int(round(box[1] * h))
        x2 = int(round(box[2] * w))
        y2 = int(round(box[3] * h))
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))

        label = int(labels[i])
        color = (0, 215, 255) if ground_truth else color_for(label)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        name = class_names[label] if label < len(class_names) else str(label)
        text = f"GT {name}" if ground_truth else name
        if scores is not None and not ground_truth:
            text = f"{name} {scores[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 4)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(
            out, text, (x1 + 1, max(th, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return out


def side_by_side(
    img_bgr: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: Sequence[int],
    pred_boxes: np.ndarray,
    pred_labels: Sequence[int],
    pred_scores: Sequence[float],
    class_names: Sequence[str] = (),
    title_left: str = "Ground Truth",
    title_right: str = "SSD300 Prediction",
) -> np.ndarray:
    """GT 与预测左右并排，便于报告里做对比图。"""
    left = draw_boxes(img_bgr, gt_boxes, gt_labels, class_names=class_names, ground_truth=True)
    right = draw_boxes(img_bgr, pred_boxes, pred_labels, pred_scores, class_names=class_names)

    h = left.shape[0]
    bar = 34
    canvas = np.full((h + bar, left.shape[1] + right.shape[1] + 4, 3), 255, dtype=np.uint8)
    canvas[bar:, : left.shape[1]] = left
    canvas[bar:, left.shape[1] + 4 :] = right
    cv2.putText(canvas, title_left, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(
        canvas, title_right, (left.shape[1] + 14, 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA,
    )
    return canvas



# --------------------------------------------------------------------------------------
# matplotlib 中文字体配置
# --------------------------------------------------------------------------------------
def setup_matplotlib():
    """配置 matplotlib 使用系统中文字体，避免图上的中文显示成方框。

    macOS 自带 PingFang SC / Heiti SC / Arial Unicode MS 等中文字体，
    这里按可用性依次尝试。
    """
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Arial Unicode MS", "Songti SC",
        "Hiragino Sans GB", "STHeiti", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False  # 正常显示负号
    matplotlib.rcParams["figure.dpi"] = 120
    import matplotlib.pyplot as plt

    return plt


# --------------------------------------------------------------------------------------
# 训练曲线
# --------------------------------------------------------------------------------------
def read_train_log(path: str) -> Dict[str, np.ndarray]:
    """读取 train_log.csv。"""
    cols: Dict[str, List[float]] = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return {k: np.array(v) for k, v in cols.items()}


def plot_training_curves(
    log_path: str,
    out_path: str,
    title: str = "SSD300 训练曲线",
    smooth: int = 5,
) -> None:
    """画 loss / 学习率 / 正样本数曲线。"""
    plt = setup_matplotlib()

    data = read_train_log(log_path)
    it = data["iter"]

    def smooth_curve(y):
        if smooth <= 1 or len(y) < smooth:
            return y
        k = np.ones(smooth) / smooth
        return np.convolve(y, k, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    ax = axes[0]
    ax.plot(it, data["loss"], alpha=0.28, color="tab:blue", label="原始")
    ax.plot(it[smooth - 1 :], smooth_curve(data["loss"]), color="tab:blue", lw=2, label=f"滑动平均({smooth})")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("总损失")
    ax.set_title("总损失")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(it, data["loc_loss"], alpha=0.25, color="tab:orange")
    ax.plot(it[smooth - 1 :], smooth_curve(data["loc_loss"]), color="tab:orange", lw=2, label="定位损失")
    ax.plot(it, data["conf_loss"], alpha=0.25, color="tab:green")
    ax.plot(it[smooth - 1 :], smooth_curve(data["conf_loss"]), color="tab:green", lw=2, label="分类损失")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("损失")
    ax.set_title("定位损失 vs 分类损失")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(it, data["lr"], color="tab:red", lw=2)
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("学习率")
    ax.set_yscale("log")
    ax.set_title("学习率调度（阶梯下降）")
    ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(
    npz_path: str,
    out_path: str,
    class_names: Sequence[str],
    ap_per_class: Optional[Dict[str, float]] = None,
) -> None:
    """画每类的 Precision-Recall 曲线。"""
    plt = setup_matplotlib()

    d = np.load(npz_path)
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(sorted(int(k.split("_")[1]) for k in d.files if k.startswith("prec_"))):
        prec = d[f"prec_{c}"]
        rec = d[f"rec_{c}"]
        if len(rec) == 0:
            continue
        name = class_names[c] if c < len(class_names) else str(c)
        label = f"{name} (AP={ap_per_class[c]*100:.1f}%)" if ap_per_class and c in ap_per_class else name
        ax.plot(np.concatenate([[0], rec, [1]]), np.concatenate([[0], prec, [0]]),
                color=cmap(i % 20), lw=1.4, label=label)
    ax.set_xlabel("召回率 Recall")
    ax.set_ylabel("精确率 Precision")
    ax.set_title("VOC2007 test 各类 PR 曲线")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_ap_bar(
    metrics_path: str,
    out_path: str,
    title: str = "SSD300 各类 AP（VOC2007 test）",
) -> None:
    """画 20 类 AP 柱状图。"""
    plt = setup_matplotlib()

    with open(metrics_path, "r") as f:
        m = json.load(f)
    aps = m["ap_per_class"]
    names = list(aps.keys())
    values = [aps[n] for n in names]

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(range(len(names)), values, color="tab:blue", alpha=0.85)
    ax.axhline(m["mAP_percent"], color="tab:red", ls="--", lw=1.8,
               label=f"mAP = {m['mAP_percent']:.2f}%")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("AP (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
