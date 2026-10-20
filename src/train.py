"""SSD300 训练脚本。

用法::

    # 快速验证（8000 迭代）
    ./env_ssd/bin/python -m src.train --config src/configs/ssd300_voc_quick.yaml

    # 断点续训
    ./env_ssd/bin/python -m src.train --config ... --resume experiments/xxx/last.pth

训练配置中的每一项都对应官方 ``examples/ssd/ssd_pascal.py``，
详见 ``src/config.py`` 中的注释与 ``代码/README_代码说明.md``。

产出（写入 ``experiments/<exp_name>/``）::

    config.yaml      本次运行的完整配置（调参留痕）
    train_log.csv    每个记录点的 loss / lr / 耗时（画曲线用）
    last.pth         最新权重（断点续训用）
    iter_XXXX.pth    周期性快照
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import describe_config, load_config
from .datasets.augment import SSDTransform
from .datasets.voc import VOCDetection, detection_collate
from .models.loss import MultiBoxLoss
from .models.prior_box import PriorBox, build_prior_box, model_head_spec
from .models.vgg_ssd import build_ssd300


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def build_loaders(cfg: Dict, seed: int):
    d = cfg["data"]
    a = cfg["augment"]
    size = cfg["model"]["input_size"]

    train_tf = SSDTransform(
        resize=size,
        mean_bgr=a["mean_bgr"],
        train=True,
        use_distort=a["use_distort"],
        use_expand=a["use_expand"],
        use_crop=a["use_crop"],
        use_mirror=a["mirror"],
        expand_prob=a["expand_prob"],
        max_expand_ratio=a["max_expand_ratio"],
        seed=seed,
        norm_mode=a.get("norm_mode", "imagenet"),
    )
    train_set = VOCDetection(d["voc_root"], d["year"], d["train_image_set"], transform=train_tf)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["solver"]["batch_size"],
        shuffle=True,
        num_workers=d["num_workers"],
        collate_fn=detection_collate,
        drop_last=True,
        persistent_workers=d["num_workers"] > 0,
    )
    return train_set, train_loader


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SSD300 训练")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="从快照继续训练")
    parser.add_argument("--max-iter", type=int, default=None, help="覆盖配置中的 max_iter")
    parser.add_argument("--exp-name", type=str, default=None, help="覆盖实验名")
    parser.add_argument("--device", type=str, default=None, help="mps / cpu / auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None,
                        help="覆盖学习率 base_lr（调参实验用）")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="覆盖 weight_decay")
    parser.add_argument("--norm-mode", type=str, default=None,
                        choices=["imagenet", "caffe"], help="覆盖输入归一化方式")
    parser.add_argument("--no-skip-nonfinite", action="store_true",
                        help="关闭梯度安全网（用于复现 lr 过大导致的梯度爆炸）")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.max_iter is not None:
        cfg["solver"]["max_iter"] = args.max_iter
    if args.exp_name:
        cfg["runtime"]["exp_name"] = args.exp_name
    if args.device:
        cfg["runtime"]["device"] = args.device
    if args.batch_size:
        cfg["solver"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.lr is not None:
        cfg["solver"]["base_lr"] = args.lr
    if args.weight_decay is not None:
        cfg["solver"]["weight_decay"] = args.weight_decay
    if args.norm_mode:
        cfg["augment"]["norm_mode"] = args.norm_mode
    if args.no_skip_nonfinite:
        cfg["runtime"]["skip_nonfinite"] = False
    # 重新推导派生参数（milestones 依赖 max_iter）
    from .config import _finalize

    _finalize(cfg)

    out_dir = os.path.join(cfg["runtime"]["output_dir"], cfg["runtime"]["exp_name"])
    os.makedirs(out_dir, exist_ok=True)

    set_seed(cfg["runtime"]["seed"])
    device = pick_device(cfg["runtime"]["device"])

    print(describe_config(cfg))
    print(f"输出目录 : {out_dir}")
    print(f"计算设备 : {device}")
    sys.stdout.flush()

    # 保存本次运行的配置（调参留痕）
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml_dump = __import__("yaml").safe_dump
        f.write(yaml_dump(cfg, allow_unicode=True, sort_keys=False))

    # ---------------- 数据 ----------------
    train_set, train_loader = build_loaders(cfg, cfg["runtime"]["seed"])
    print(f"训练集   : {len(train_set)} 张 ({cfg['data']['year']} {cfg['data']['train_image_set']})")

    # ---------------- 模型 ----------------
    prior_box = build_prior_box(cfg)
    head_spec = model_head_spec(cfg)
    priors = prior_box().to(device)
    variances = prior_box.variances.to(device)
    print(
        f"先验框   : {priors.shape[0]} 个"
        f"  (预测层 {list(head_spec['prediction_layers'])}，"
        f"每层框数 {list(head_spec['num_priors_per_layer'])})"
    )

    model = build_ssd300(
        num_classes=cfg["data"]["num_classes"],
        pretrained=cfg["model"]["pretrained_backbone"],
        backbone_weights=cfg["model"].get("backbone_weights", ""),
        **head_spec,
    ).to(device)

    criterion = MultiBoxLoss(
        priors=priors,
        variances=variances,
        num_classes=cfg["data"]["num_classes"],
        neg_pos_ratio=cfg["loss"]["neg_pos_ratio"],
        overlap_threshold=cfg["loss"]["overlap_threshold"],
        neg_overlap=cfg["loss"]["neg_overlap"],
        background_label_id=cfg["loss"]["background_label_id"],
        loc_weight=cfg["loss"]["loc_weight"],
        mining_type=cfg["loss"].get("mining_type", "max_negative"),
        normalize_conf_by_samples=cfg["loss"].get("normalize_conf_by_samples", False),
    ).to(device)

    # ---------------- 优化器与学习率 ----------------
    # 对齐官方 ConvBNLayer 的参数设置（use_batchnorm=False 分支）：
    #     param: [dict(lr_mult=1, decay_mult=1),    # 权重：学习率 x1，做 weight decay
    #             dict(lr_mult=2, decay_mult=0)]    # 偏置：学习率 x2，**不做** weight decay
    s = cfg["solver"]
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias"):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = optim.SGD(
        [
            {"params": decay_params, "lr": s["base_lr"] * 1.0, "weight_decay": s["weight_decay"]},
            {"params": no_decay_params, "lr": s["base_lr"] * 2.0, "weight_decay": 0.0},
        ],
        momentum=s["momentum"],
    )
    print(
        f"优化器   : SGD(momentum={s['momentum']})  "
        f"权重组={len(decay_params)} 个张量(lr={s['base_lr']:.1e}, wd={s['weight_decay']}), "
        f"偏置组={len(no_decay_params)} 个张量(lr={s['base_lr']*2:.1e}, wd=0)"
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=s["milestones"], gamma=s["gamma"]
    )

    start_iter = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_iter = ckpt["iter"]
        print(f"从快照恢复: {args.resume} (已训练 {start_iter} 迭代)")

    # ---------------- 训练循环 ----------------
    log_path = os.path.join(out_dir, "train_log.csv")
    log_is_new = not os.path.isfile(log_path)
    log_f = open(log_path, "a", newline="")
    log_w = csv.writer(log_f)
    if log_is_new:
        log_w.writerow(["iter", "loss", "loc_loss", "conf_loss", "num_pos", "lr", "elapsed_sec"])

    max_iter = int(s["max_iter"])
    grad_clip = float(cfg["runtime"].get("grad_clip", 0.0))
    skip_nonfinite = bool(cfg["runtime"].get("skip_nonfinite", True))
    max_grad_norm = float(cfg["runtime"].get("max_grad_norm", 2000.0))
    skipped = 0
    abort = False
    diverged = False
    model.train()
    t0 = time.time()
    running = {"loss": 0.0, "loc": 0.0, "conf": 0.0, "pos": 0, "n": 0}
    it = start_iter
    pbar = tqdm(total=max_iter, initial=start_iter, desc="训练", unit="iter", ncols=110)

    # 官方 solver 里配了 max_iter，且上限不超过 100000，这里用 assert 提醒
    while it < max_iter:
        for images, boxes, labels, _, _, _ in train_loader:
            if it >= max_iter:
                break
            images = images.to(device, non_blocking=True)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            loc_pred, conf_pred = model(images)
            loss, stats = criterion(loc_pred, conf_pred, boxes, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # ---- 安全网：梯度异常时跳过该步，保护长时间训练（正常训练不会触发）----
            if skip_nonfinite or grad_clip > 0:
                gnorm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip if grad_clip > 0 else 1e12
                )
                bad = (not torch.isfinite(loss)) or (not torch.isfinite(gnorm))
                if skip_nonfinite and float(gnorm) > max_grad_norm:
                    bad = True
                if bad:
                    skipped += 1
                    optimizer.zero_grad(set_to_none=True)
                    if skipped <= 20 or skipped % 100 == 0:
                        tqdm.write(
                            f"  [警告] 第 {it + 1} 迭代梯度异常"
                            f"(loss={float(loss.detach()):.3g}, |grad|={float(gnorm):.3g})，已跳过；"
                            f"累计跳过 {skipped} 次"
                        )
                    if skipped > 200:
                        tqdm.write("  [中止] 异常次数过多，停止训练并保存当前权重。")
                        abort = True
                        break
                    it += 1
                    pbar.update(1)
                    continue

            if not skip_nonfinite and grad_clip <= 0:
                # 安全网已关闭：只做发散检测，一旦 loss 非有限值就立即停止
                if not torch.isfinite(loss):
                    lr_now = optimizer.param_groups[0]["lr"]
                    pbar.close()   # 先关进度条，否则崩溃提示会被进度条切碎
                    tqdm.write("")
                    tqdm.write("=" * 66)
                    tqdm.write(f"  [崩溃] 第 {it + 1} 迭代 loss = {float(loss.detach())}")
                    tqdm.write("         训练已发散，自动停止（未保存可用权重）。")
                    tqdm.write(f"         当前学习率 lr = {lr_now:.2g}")
                    tqdm.write("         典型原因：学习率过大导致梯度爆炸。")
                    tqdm.write(f"         参考：本实验 lr={lr_now:.2g}，正常配置为 5e-4。")
                    tqdm.write("=" * 66)
                    log_w.writerow([it + 1, "nan", "nan", "nan",
                                    running["pos"] / max(running["n"], 1),
                                    f"{lr_now:.8g}", f"{time.time() - t0:.1f}"])
                    log_f.flush()
                    _save(os.path.join(out_dir, "diverged.pth"),
                          model, optimizer, scheduler, it + 1, cfg)
                    diverged = True
                    abort = True
                    break

            optimizer.step()
            scheduler.step()

            running["loss"] += stats["loss"]
            running["loc"] += stats["loc_loss"]
            running["conf"] += stats["conf_loss"]
            running["pos"] += stats["num_pos"]
            running["n"] += 1
            it += 1
            pbar.update(1)

            if it % cfg["runtime"]["log_interval"] == 0 or it == max_iter:
                n = max(running["n"], 1)
                elapsed = time.time() - t0
                lr_now = optimizer.param_groups[0]["lr"]
                log_w.writerow(
                    [
                        it,
                        f"{running['loss'] / n:.6f}",
                        f"{running['loc'] / n:.6f}",
                        f"{running['conf'] / n:.6f}",
                        f"{running['pos'] / n:.1f}",
                        f"{lr_now:.8g}",
                        f"{elapsed:.1f}",
                    ]
                )
                log_f.flush()
                pbar.set_postfix(
                    loss=f"{running['loss'] / n:.4f}",
                    loc=f"{running['loc'] / n:.4f}",
                    conf=f"{running['conf'] / n:.4f}",
                    pos=f"{running['pos'] / n:.1f}",
                    lr=f"{lr_now:.1e}",
                )
                running = {"loss": 0.0, "loc": 0.0, "conf": 0.0, "pos": 0, "n": 0}

            if it % cfg["runtime"]["snapshot_interval"] == 0:
                _save(os.path.join(out_dir, f"iter_{it}.pth"), model, optimizer, scheduler, it, cfg)
                _save(os.path.join(out_dir, "last.pth"), model, optimizer, scheduler, it, cfg)
        if abort:
            break

    if not diverged:
        _save(os.path.join(out_dir, "last.pth"), model, optimizer, scheduler, it, cfg)
    pbar.close()
    log_f.close()

    total = time.time() - t0
    summary = {
        "exp_name": cfg["runtime"]["exp_name"],
        "iters": it,
        "total_seconds": round(total, 1),
        "seconds_per_iter": round(total / max(it - start_iter, 1), 3),
        "device": str(device),
        "batch_size": s["batch_size"],
        "milestones": s["milestones"],
        "base_lr": s["base_lr"],
        "skipped_steps": skipped,
        "diverged": diverged,
        "checkpoint": os.path.join(
            out_dir, "diverged.pth" if diverged else "last.pth"
        ),
    }
    with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n训练完成")
    for k, v in summary.items():
        print(f"    {k:20s} = {v}")
    return 0


def _save(path, model, optimizer, scheduler, it, cfg):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "iter": it,
            "config": cfg,
        },
        path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
