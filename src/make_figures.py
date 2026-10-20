"""生成报告所需的全部图表。

用法::

    ./env_ssd/bin/python -m src.make_figures --exp-name quick8000

产出 ``experiments/<exp_name>/figures/``::

    training_curves.png       训练损失 / 学习率曲线
    ap_bar.png                20 类 AP 柱状图
    pr_curves.png             各类 PR 曲线
    detections/*.png          GT vs 预测 对比图（可直接插入报告）
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import cv2
import numpy as np
import torch

from .config import load_config
from .datasets.augment import SSDTransform
from .datasets.voc import VOC_CLASSES, VOCDetection
from .models.detection import detect_batch
from .models.prior_box import build_prior_box, model_head_spec
from .models.vgg_ssd import build_ssd300
from .train import pick_device
from .utils.io_utils import imread_unicode, imwrite_unicode
from .utils.visualize import (
    plot_ap_bar,
    plot_pr_curves,
    plot_training_curves,
    side_by_side,
)


def make_detection_figures(
    cfg: Dict,
    weights: str,
    device: torch.device,
    out_dir: str,
    num_images: int = 8,
    display_threshold: float = 0.30,
    max_draw: int = 25,
) -> List[str]:
    """在若干测试图上画「GT vs 预测」对比图。"""
    d, a, det = cfg["data"], cfg["augment"], cfg["detection"]
    size = cfg["model"]["input_size"]

    test_tf = SSDTransform(
        resize=size, mean_bgr=a["mean_bgr"], train=False,
        norm_mode=a.get("norm_mode", "imagenet"),
    )
    test_set = VOCDetection(d["voc_root"], d["year"], d["test_image_set"], transform=test_tf)

    prior_box = build_prior_box(cfg)
    priors = prior_box().to(device)
    variances = prior_box.variances.to(device)

    model = build_ssd300(
        num_classes=d["num_classes"], pretrained=False, **model_head_spec(cfg)
    ).to(device)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 挑选目标数量适中的图（太少的图没信息量，太多的图看不清）
    counts = []
    for i in range(min(600, len(test_set))):
        boxes, labels, _ = test_set._load_annotation(test_set.ids[i])
        counts.append((i, len(labels)))
    counts.sort(key=lambda t: t[1])
    mid = len(counts) // 2
    chosen = [counts[mid + k * 7][0] for k in range(num_images) if mid + k * 7 < len(counts)]

    os.makedirs(out_dir, exist_ok=True)
    saved = []
    with torch.no_grad():
        for rank, idx in enumerate(chosen):
            sample = test_set[idx]
            img_id = test_set.ids[idx]
            img_t = sample["image"].unsqueeze(0).to(device)
            loc_pred, conf_pred = model(img_t)
            cls, score, box = detect_batch(
                loc_pred, conf_pred, priors, variances,
                num_classes=d["num_classes"],
                background_label_id=cfg["loss"]["background_label_id"],
                confidence_threshold=det["confidence_threshold"],
                nms_threshold=det["nms_threshold"],
                top_k=det["top_k"],
                keep_top_k=det["keep_top_k"],
            )[0]

            # 读原图（BGR）
            raw = imread_unicode(os.path.join(test_set.root, "JPEGImages", f"{img_id}.jpg"))
            gt_boxes, gt_labels, _ = test_set._load_annotation(img_id)

            # 为可读性只画得分较高的前 max_draw 个框（原始输出有 keep_top_k=200 个）
            sc = score.cpu().numpy()
            keep = np.where(sc >= display_threshold)[0]
            if keep.size == 0:
                keep = np.argsort(-sc)[:5]          # 一个都不到阈值时，至少画前 5 个
            keep = keep[np.argsort(-sc[keep])][:max_draw]
            canvas = side_by_side(
                raw,
                np.asarray(gt_boxes, dtype=np.float64),
                gt_labels,
                box.cpu().numpy().astype(np.float64)[keep],
                np.asarray(cls.cpu().numpy())[keep].tolist(),
                sc[keep].tolist(),
                class_names=VOC_CLASSES,
            )
            path = os.path.join(out_dir, f"{rank:02d}_{img_id}.png")
            imwrite_unicode(path, canvas)
            saved.append(path)
            print(
                f"    {img_id}: GT {len(gt_labels)} 个目标, 预测 {cls.numel()} 个框"
            )
    return saved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成报告图表")
    parser.add_argument("--exp-name", type=str, default="quick8000")
    parser.add_argument("--config", type=str, default="src/configs/ssd300_voc_quick.yaml")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--skip-detections", action="store_true")
    parser.add_argument("--display-threshold", type=float, default=0.30,
                        help="可视化时只画得分高于该值的检测框")
    parser.add_argument("--max-draw", type=int, default=25, help="每图最多画多少个框")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    exp_dir = os.path.join(cfg["runtime"]["output_dir"], args.exp_name)
    fig_dir = os.path.join(exp_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"实验目录 : {exp_dir}")
    print(f"图表目录 : {fig_dir}")

    # ---- ① 训练曲线 ----
    log_path = os.path.join(exp_dir, "train_log.csv")
    if os.path.isfile(log_path):
        out = os.path.join(fig_dir, "training_curves.png")
        plot_training_curves(log_path, out, title=f"SSD300 训练曲线（{args.exp_name}）")
        print(f"  [完成] {out}")
    else:
        print(f"  [跳过] 找不到 {log_path}")

    # ---- ② AP 柱状图 ----
    metrics_path = os.path.join(exp_dir, "eval", "metrics.json")
    if os.path.isfile(metrics_path):
        out = os.path.join(fig_dir, "ap_bar.png")
        plot_ap_bar(metrics_path, out, title=f"SSD300 各类 AP（VOC2007 test, {args.exp_name}）")
        print(f"  [完成] {out}")

        # ---- ③ PR 曲线 ----
        npz_path = os.path.join(exp_dir, "eval", "pr_curves.npz")
        if os.path.isfile(npz_path):
            with open(metrics_path) as f:
                m = json.load(f)
            ap_map = {i: v / 100.0 for i, v in enumerate(m["ap_per_class"].values(), start=1)}
            out = os.path.join(fig_dir, "pr_curves.png")
            plot_pr_curves(npz_path, out, VOC_CLASSES, ap_per_class=ap_map)
            print(f"  [完成] {out}")
    else:
        print(f"  [跳过] 找不到 {metrics_path}（请先跑评测）")

    # ---- ④ 检测结果可视化 ----
    weights = args.weights or os.path.join(exp_dir, "last.pth")
    if not args.skip_detections and os.path.isfile(weights):
        device = pick_device(cfg["runtime"]["device"])
        print(f"  生成检测可视化（设备 {device}，权重 {weights}）...")
        saved = make_detection_figures(
            cfg, weights, device, os.path.join(fig_dir, "detections"), args.num_images,
            display_threshold=args.display_threshold, max_draw=args.max_draw,
        )
        print(f"  [完成] 共 {len(saved)} 张检测对比图")
    else:
        print("  [跳过] 检测可视化")

    print("\n全部图表已生成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
