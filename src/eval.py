"""SSD300 评测脚本：在 VOC2007 test 上计算 mAP。

用法::

    ./env_ssd/bin/python -m src.eval \
        --config src/configs/ssd300_voc_quick.yaml \
        --weights experiments/quick8000/last.pth

产出（写入 ``experiments/<exp_name>/eval/``）::

    metrics.json          mAP 与 20 类逐类 AP
    pr_curves.npz         每类的 precision / recall 曲线（画 PR 图用）
    detections.json       每张图的检测结果（可视化与错误分析用）
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .datasets.augment import SSDTransform
from .datasets.voc import VOC_CLASSES, VOCDetection, detection_collate
from .models.detection import detect_batch
from .models.prior_box import PriorBox, build_prior_box, model_head_spec
from .models.vgg_ssd import build_ssd300
from .train import pick_device
from .utils.voc_eval import evaluate_detections


@torch.no_grad()
def run_eval(cfg: Dict, weights: str, device: torch.device, max_images: int = 0) -> Dict:
    d, a, det, ev = cfg["data"], cfg["augment"], cfg["detection"], cfg["eval"]
    size = cfg["model"]["input_size"]

    test_tf = SSDTransform(
        resize=size, mean_bgr=a["mean_bgr"], train=False,
        norm_mode=a.get("norm_mode", "imagenet"),
    )
    test_set = VOCDetection(d["voc_root"], d["year"], d["test_image_set"], transform=test_tf)
    if max_images:
        test_set.ids = test_set.ids[:max_images]
    print(f"评测集   : {len(test_set)} 张 ({d['year']} {d['test_image_set']})")

    test_loader = DataLoader(
        test_set,
        batch_size=ev["batch_size"],
        shuffle=False,
        num_workers=d["num_workers"],
        collate_fn=detection_collate,
    )

    # 权重文件不存在时，给出可用快照列表而不是抛原始 FileNotFoundError
    if not os.path.isfile(weights):
        print(f"\n❌ 找不到权重文件: {weights}")
        import glob as _glob
        cands = sorted(_glob.glob(os.path.join("experiments", "*", "*.pth")))
        if cands:
            print("   可用的权重文件：")
            for c in cands:
                print(f"     {c}  ({os.path.getsize(c)/1e6:.0f} MB)")
        else:
            print("   experiments/ 下没有任何 .pth 文件，请先训练。")
        raise SystemExit(1)

    prior_box = build_prior_box(cfg)
    priors = prior_box().to(device)
    variances = prior_box.variances.to(device)

    model = build_ssd300(
        num_classes=d["num_classes"], pretrained=False, **model_head_spec(cfg)
    ).to(device)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"权重     : {weights} (第 {ckpt.get('iter', '?')} 迭代)")

    all_dets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    all_gts: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    per_image: List[Dict] = []

    t0 = time.time()
    for images, gt_boxes, gt_labels, ids, orig_sizes, difficults in tqdm(
        test_loader, desc="推理", ncols=100
    ):
        images = images.to(device)
        loc_pred, conf_pred = model(images)
        results = detect_batch(
            loc_pred,
            conf_pred,
            priors,
            variances,
            num_classes=d["num_classes"],
            background_label_id=cfg["loss"]["background_label_id"],
            confidence_threshold=det["confidence_threshold"],
            nms_threshold=det["nms_threshold"],
            top_k=det["top_k"],
            keep_top_k=det["keep_top_k"],
        )
        # detect_batch 返回「展平」的列表：每个元素对应一张图
        for b in range(images.shape[0]):
            cls, score, box = results[b]
            all_dets.append(
                (
                    cls.cpu().numpy(),
                    score.cpu().numpy(),
                    box.cpu().numpy(),
                )
            )
            per_image.append(
                {
                    "id": ids[b],
                    "labels": cls.cpu().numpy().tolist(),
                    "scores": [round(float(s), 5) for s in score.cpu().numpy()],
                    "boxes": [[round(float(v), 5) for v in bx] for bx in box.cpu().numpy()],
                }
            )

            # 真值直接用 DataLoader 返回的（测试变换不改动框）
            all_gts.append(
                (
                    gt_boxes[b].numpy().astype(np.float64),
                    gt_labels[b].numpy().astype(np.int64),
                    difficults[b].numpy().astype(np.int64),
                )
            )

    elapsed = time.time() - t0
    fps = len(test_set) / elapsed
    print(f"推理完成 : {elapsed:.1f} 秒  ({fps:.2f} FPS, batch={ev['batch_size']})")

    metrics = evaluate_detections(
        all_dets,
        all_gts,
        num_classes=d["num_classes"],
        background_label_id=cfg["loss"]["background_label_id"],
        overlap_threshold=ev["overlap_threshold"],
        evaluate_difficult_gt=ev["evaluate_difficult_gt"],
        class_names=VOC_CLASSES,
    )
    metrics["fps"] = fps
    metrics["elapsed_sec"] = elapsed
    metrics["num_images"] = len(test_set)
    metrics["weights"] = weights
    metrics["iter"] = ckpt.get("iter")
    metrics["_per_image"] = per_image
    return metrics


def _load_gt(dataset: VOCDetection, img_id: str):
    """读取某张图像的原始真值标注（不做增强）。"""
    boxes, labels, difficult = dataset._load_annotation(img_id)
    return (
        np.asarray(boxes, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(difficult, dtype=np.int64),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SSD300 评测")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=0, help="只评测前 N 张（调试用）")
    parser.add_argument("--out", type=str, default=None, help="结果输出目录")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="覆盖 DataLoader 进程数；容器 /dev/shm 小时用 0")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.device:
        cfg["runtime"]["device"] = args.device
    device = pick_device(cfg["runtime"]["device"])
    print(f"计算设备 : {device}")

    metrics = run_eval(cfg, args.weights, device, max_images=args.max_images)
    per_image = metrics.pop("_per_image", [])

    if args.out:
        out_dir = args.out
    else:
        # 默认输出到「权重所在目录」下的 eval/，
        # 而不是配置里的 exp_name —— 否则评测 cloud5000 的权重时，
        # 结果会被写进 experiments/quick8000/eval/，很容易找错地方。
        _wdir = os.path.dirname(os.path.abspath(args.weights))
        if not os.path.isdir(_wdir):
            _wdir = os.path.join(cfg["runtime"]["output_dir"], cfg["runtime"]["exp_name"])
        out_dir = os.path.join(_wdir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    # 把 AP 表打印出来（可直接截图进报告）
    names = VOC_CLASSES
    print("\n" + "=" * 62)
    print(f"{'类别':<14}{'AP':>10}")
    print("-" * 62)
    for c in range(1, len(names)):
        ap = metrics["ap_per_class"].get(c, 0.0)
        print(f"{names[c]:<14}{ap * 100:>9.2f}%")
    print("-" * 62)
    print(f"{'mAP':<14}{metrics['mAP'] * 100:>9.2f}%")
    print("=" * 62)

    serializable = {
        "mAP": metrics["mAP"],
        "mAP_percent": round(metrics["mAP"] * 100, 2),
        "ap_per_class": {names[c]: round(metrics["ap_per_class"].get(c, 0.0) * 100, 2)
                         for c in range(1, len(names))},
        "fps": round(metrics["fps"], 3),
        "elapsed_sec": round(metrics["elapsed_sec"], 1),
        "num_images": metrics["num_images"],
        "weights": metrics["weights"],
        "iter": metrics["iter"],
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    np.savez(
        os.path.join(out_dir, "pr_curves.npz"),
        **{f"prec_{c}": metrics["pr_curves"][c][0] for c in metrics["pr_curves"]},
        **{f"rec_{c}": metrics["pr_curves"][c][1] for c in metrics["pr_curves"]},
    )

    # 每张图的检测结果（可视化与错误分析用）
    with open(os.path.join(out_dir, "detections.json"), "w") as f:
        json.dump(per_image, f, ensure_ascii=False)

    print(f"\n结果已保存到: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
