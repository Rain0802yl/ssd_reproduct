"""配置定义与加载。

默认值全部来自官方 ``examples/ssd/ssd_pascal.py``，逐项标注了出处，
便于在报告中说明「哪些是官方原样，哪些是为了适配本机算力而调整的」。
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import yaml


# --------------------------------------------------------------------------------------
# 官方 SSD300 + VOC2007 的默认配置
# --------------------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # ---------------- 数据 ----------------
    "data": {
        "voc_root": "数据集/VOCdevkit",      # 包含 VOC2007 的目录
        "year": "2007",
        "train_image_set": "trainval",        # 官方 07 设定
        "test_image_set": "test",
        "num_classes": 21,                    # 官方 num_classes
        "num_workers": 4,
    },
    # ---------------- 模型 ----------------
    "model": {
        "input_size": 300,                    # 官方 min_dim = 300
        "pretrained_backbone": True,          # 使用 ImageNet 预训练 VGG16
        # 本地权重文件（留空则用 torchvision 自动下载）；指向工作区缓存可离线运行
        "backbone_weights": ".cache/torch/hub/checkpoints/vgg16_features-amdegroot-88682ab5.pth",
        "l2_norm_scale": 20.0,                # 官方 normalizations[0] = 20
        # 官方 min_sizes / max_sizes（由 min_ratio=20, max_ratio=90, step=17 推出）
        "min_sizes": [30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
        "max_sizes": [60.0, 111.0, 162.0, 213.0, 264.0, 315.0],
        "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        "steps": [8, 16, 32, 64, 100, 300],
        "variances": [0.1, 0.1, 0.2, 0.2],    # 官方 prior_variance
        "clip_prior": False,                  # 官方 clip = False
        "flip_priors": True,                  # 官方 flip = True（为每个长宽比补上倒数）
        # 用哪些层做预测（0=conv4_3, 1=conv7, 2=conv8_2, 3=conv9_2, 4=conv10_2, 5=conv11_2）
        # 默认 6 层全用；消融实验可只取子集，网络头与先验框会自动保持一致
        "prediction_layers": [0, 1, 2, 3, 4, 5],
    },
    # ---------------- 数据增强（官方 train_transform_param + batch_sampler）----------------
    "augment": {
        # 输入归一化方式：
        #   "imagenet" = (x/255 - ImageNet均值)/ImageNet标准差（匹配 torchvision VGG16 权重）
        #   "caffe"    = BGR 直接减 [104,117,123]（官方 Caffe 约定）
        "norm_mode": "imagenet",
        "mean_bgr": [104, 117, 123],          # 官方 mean_value（norm_mode="caffe" 时生效）
        "mirror": True,
        "use_distort": True,                  # 消融开关：光度扰动
        "use_expand": True,                   # 消融开关：随机扩展
        "use_crop": True,                     # 消融开关：随机裁剪
        "expand_prob": 0.5,
        "max_expand_ratio": 4.0,
    },
    # ---------------- 损失（官方 multibox_loss_param）----------------
    "loss": {
        "neg_pos_ratio": 3.0,
        "overlap_threshold": 0.5,
        "neg_overlap": 0.5,
        "background_label_id": 0,
        "loc_weight": 1.0,                    # 官方 loc_weight = (3+1)/4 = 1.0
        # 官方 mining_type = MAX_NEGATIVE；消融实验可设为 "none"（关闭难负样本挖掘）
        "mining_type": "max_negative",
        # 分类损失是否按「分类样本数」而不是「正样本数」归一化。
        # 官方为 False（VALID 归一化）。关闭挖掘时负样本暴增，需设为 True 才能保持量级。
        "normalize_conf_by_samples": False,
    },
    # ---------------- 优化器（官方 solver_param）----------------
    "solver": {
        "base_lr": 0.001,                     # 官方 0.00004 * 25/loc_weight = 0.001（对应 batch=32）
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "lr_policy": "multistep",
        "gamma": 0.1,
        # 官方官方里程碑为 [80000, 100000, 120000]，max_iter = 120000
        "milestones": [80000, 100000, 120000],
        "max_iter": 120000,
        "batch_size": 32,
        # 为了让短训练也能走完完整的阶梯下降，可把里程碑按比例缩放到 max_iter
        "scale_milestones": False,
        # 说明：官方 base_lr = 1e-3 是对应 batch_size = 32 的。
        # 本机内存限制导致 batch 只能开到 16，按「线性缩放规则」学习率同步减半为 5e-4。
        # 实测：batch=16 时 lr=1e-3 会在第 9 个迭代梯度爆炸成 NaN；lr=5e-4 可稳定训练。
    },
    # ---------------- 推理（官方 det_out_param）----------------
    "detection": {
        "confidence_threshold": 0.01,
        "nms_threshold": 0.45,
        "top_k": 400,
        "keep_top_k": 200,
    },
    # ---------------- 评测 ----------------
    "eval": {
        "batch_size": 8,                      # 官方 test_batch_size = 8
        "overlap_threshold": 0.5,
        "evaluate_difficult_gt": False,       # 官方 evaluate_difficult_gt = False
        "ap_version": "11point",              # 官方 ap_version = 11point
        "test_interval": 10000,               # 官方 test_interval
    },
    # ---------------- 运行 ----------------
    "runtime": {
        "output_dir": "experiments",
        "exp_name": "quick8000",
        "device": "auto",                     # auto / mps / cpu
        "log_interval": 10,                   # 官方 display = 10
        "snapshot_interval": 2000,
        "seed": 0,
        "resume": None,
        # ---- 以下为长时间训练的「安全网」，正常训练时不会触发 ----
        # 官方原版没有梯度裁剪；这里仅在梯度异常爆炸时兜底，避免数小时训练白跑。
        "grad_clip": 0.0,                     # >0 时按梯度总范数裁剪；0 = 关闭（默认关闭）
        "skip_nonfinite": True,               # loss/梯度出现 NaN 或 Inf 时跳过该步，不更新权重
        "max_grad_norm": 2000.0,              # 梯度总范数超过该值也跳过该步
                                              # （实测正常训练中位数 7.6、最大 102，故正常不会触发）
    },
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Optional[str] = None, **overrides: Any) -> Dict[str, Any]:
    """加载配置：默认值 <- YAML 文件（可多层继承）<- 关键字参数。

    YAML 里可以用 ``_base: 另一个配置文件的路径`` 继承一份基础配置，
    只需写出要覆盖的字段。消融实验配置就是靠这个机制保持简短的。

    例::

        _base: ssd300_voc_quick.yaml
        augment:
          use_distort: false
          use_expand: false
          use_crop: false
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    user_cfg = _load_yaml_with_base(path) if path else {}
    _deep_update(cfg, user_cfg)
    if overrides:
        _deep_update(cfg, overrides)
    _finalize(cfg)
    return cfg


def _load_yaml_with_base(path: str, _seen: Optional[set] = None) -> Dict[str, Any]:
    """读取 YAML，若含 ``_base`` 则先递归加载基础配置再覆盖。"""
    _seen = _seen or set()
    path = os.path.abspath(path)
    if path in _seen:
        raise ValueError(f"配置继承出现循环: {path}")
    _seen.add(path)

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    base_ref = raw.pop("_base", None)
    if not base_ref:
        return raw

    # _base 用相对当前文件的路径解析
    base_path = base_ref
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.path.dirname(path), base_ref)
    base_cfg = _load_yaml_with_base(base_path, _seen)
    _deep_update(base_cfg, raw)
    return base_cfg


def _finalize(cfg: Dict[str, Any]) -> None:
    """按官方脚本的逻辑推导派生参数。"""
    solver = cfg["solver"]
    max_iter = int(solver["max_iter"])
    if solver.get("scale_milestones"):
        # 把官方的 [80k, 100k, 120k] 按同一比例缩放到 max_iter
        official_total = 120000
        solver["milestones"] = [
            max(1, int(round(m / official_total * max_iter))) for m in [80000, 100000, 120000]
        ]
        # 去掉重复且不超过 max_iter
        ms = sorted({m for m in solver["milestones"] if m <= max_iter})
        solver["milestones"] = ms
    else:
        solver["milestones"] = sorted({int(m) for m in solver["milestones"]})


def describe_config(cfg: Dict[str, Any]) -> str:
    """生成配置摘要，便于写入日志与报告。"""
    lines = []
    lines.append("=" * 78)
    lines.append("SSD300 复现配置")
    lines.append("=" * 78)
    for section in ["data", "model", "augment", "loss", "solver", "detection", "eval", "runtime"]:
        if section not in cfg:
            continue
        lines.append(f"[{section}]")
        for k, v in cfg[section].items():
            lines.append(f"    {k:26s} = {v}")
    lines.append("=" * 78)
    return "\n".join(lines)
