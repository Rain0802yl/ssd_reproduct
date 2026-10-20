"""边界框工具：IoU、编码/解码、裁剪、面积计算。

**全部对齐官方 Caffe 实现**，对应关系：

    jaccard_overlap / intersect_bbox  ->  src/caffe/util/bbox_util.cpp : JaccardOverlap / IntersectBBox
    bbox_size                         ->  src/caffe/util/bbox_util.cpp : BBoxSize
    encode_bbox                       ->  src/caffe/util/bbox_util.cpp : EncodeBBox  (CENTER_SIZE)
    decode_bbox                       ->  src/caffe/util/bbox_util.cpp : DecodeBBox  (CENTER_SIZE)
    clip_bbox                         ->  src/caffe/util/bbox_util.cpp : ClipBBox

所有框统一使用 **归一化角点坐标** [xmin, ymin, xmax, ymax]，取值范围 [0, 1]，
与官方代码的 NormalizedBBox 完全一致。
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------------------------
# 面积与 IoU
# --------------------------------------------------------------------------------------
def bbox_size(box: torch.Tensor) -> torch.Tensor:
    """框面积。对应 Caffe ``BBoxSize``：若 xmax<xmin 或 ymax<ymin 则返回 0。"""
    w = (box[..., 2] - box[..., 0]).clamp(min=0)
    h = (box[..., 3] - box[..., 1]).clamp(min=0)
    return w * h


def intersect_bbox(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """两两求交集框。对应 Caffe ``IntersectBBox``。

    参数形状均为 (..., 4)，返回广播后的 (..., 4)。
    无交集时按 Caffe 的做法返回退化的零框。
    """
    xmin = torch.maximum(box1[..., 0], box2[..., 0])
    ymin = torch.maximum(box1[..., 1], box2[..., 1])
    xmax = torch.minimum(box1[..., 2], box2[..., 2])
    ymax = torch.minimum(box1[..., 3], box2[..., 3])
    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


def jaccard_overlap(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """两两 IoU。对应 Caffe ``JaccardOverlap``。

    参数:
        boxes1: (N, 4)
        boxes2: (M, 4)
    返回:
        (N, M) 的 IoU 矩阵

    Caffe 的定义为 ``intersect / (size1 + size2 - intersect)``，
    且当交集面积为 0 时直接返回 0。
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    b1 = boxes1.unsqueeze(1)  # (N, 1, 4)
    b2 = boxes2.unsqueeze(0)  # (1, M, 4)

    inter = intersect_bbox(b1, b2)
    inter_w = (inter[..., 2] - inter[..., 0]).clamp(min=0)
    inter_h = (inter[..., 3] - inter[..., 1]).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = bbox_size(b1)  # (N, 1)
    area2 = bbox_size(b2)  # (1, M)
    union = area1 + area2 - inter_area

    iou = inter_area / union.clamp(min=1e-12)
    # 交集为 0 时强制返回 0（与 Caffe 一致，避免 0/0）
    iou = torch.where(inter_area <= 0, torch.zeros_like(iou), iou)
    return iou


# --------------------------------------------------------------------------------------
# 裁剪
# --------------------------------------------------------------------------------------
def clip_bbox(box: torch.Tensor) -> torch.Tensor:
    """把框裁剪到 [0, 1]。对应 Caffe ``ClipBBox``（会先判断框是否合法）。"""
    # Caffe 的 ClipBBox 只对合法框（xmax>=xmin 且 ymax>=ymin）做裁剪
    valid = (box[..., 2] >= box[..., 0]) & (box[..., 3] >= box[..., 1])
    clipped = torch.stack(
        [
            box[..., 0].clamp(0.0, 1.0),
            box[..., 1].clamp(0.0, 1.0),
            box[..., 2].clamp(0.0, 1.0),
            box[..., 3].clamp(0.0, 1.0),
        ],
        dim=-1,
    )
    return torch.where(valid.unsqueeze(-1), clipped, box)


# --------------------------------------------------------------------------------------
# 编码 / 解码（CENTER_SIZE + variance）
# --------------------------------------------------------------------------------------
def encode_bbox(prior: torch.Tensor, gt: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    """把真值框编码成相对先验框的偏移量。

    对应 Caffe ``EncodeBBox`` 中 ``code_type == CENTER_SIZE`` 且
    ``encode_variance_in_target == False`` 的分支：

        tx = (gcx - pcx) / pw / variance[0]
        ty = (gcy - pcy) / ph / variance[1]
        tw = log(gw / pw)     / variance[2]
        th = log(gh / ph)     / variance[3]
    """
    pw = prior[..., 2] - prior[..., 0]
    ph = prior[..., 3] - prior[..., 1]
    pcx = (prior[..., 0] + prior[..., 2]) / 2.0
    pcy = (prior[..., 1] + prior[..., 3]) / 2.0

    gw = (gt[..., 2] - gt[..., 0]).clamp(min=1e-12)
    gh = (gt[..., 3] - gt[..., 1]).clamp(min=1e-12)
    gcx = (gt[..., 0] + gt[..., 2]) / 2.0
    gcy = (gt[..., 1] + gt[..., 3]) / 2.0

    # 用 variance[..., k] 而不是 variance[k]，这样 (4,) 和 (...,4) 两种形状都能正确广播
    return torch.stack(
        [
            (gcx - pcx) / pw / variance[..., 0],
            (gcy - pcy) / ph / variance[..., 1],
            torch.log(gw / pw) / variance[..., 2],
            torch.log(gh / ph) / variance[..., 3],
        ],
        dim=-1,
    )


def decode_bbox(prior: torch.Tensor, loc: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    """把网络预测的偏移量解码成真实框。

    对应 Caffe ``DecodeBBox`` 中 ``code_type == CENTER_SIZE`` 且
    ``variance_encoded_in_target == False`` 的分支。
    """
    pw = prior[..., 2] - prior[..., 0]
    ph = prior[..., 3] - prior[..., 1]
    pcx = (prior[..., 0] + prior[..., 2]) / 2.0
    pcy = (prior[..., 1] + prior[..., 3]) / 2.0

    cx = variance[..., 0] * loc[..., 0] * pw + pcx
    cy = variance[..., 1] * loc[..., 1] * ph + pcy
    w = torch.exp(variance[..., 2] * loc[..., 2]) * pw
    h = torch.exp(variance[..., 3] * loc[..., 3]) * ph

    return torch.stack(
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dim=-1
    )


# --------------------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------------------
def xywh_to_corner(box: torch.Tensor) -> torch.Tensor:
    """(cx, cy, w, h) -> (xmin, ymin, xmax, ymax)，均为归一化坐标。"""
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def corner_to_xywh(box: torch.Tensor) -> torch.Tensor:
    """(xmin, ymin, xmax, ymax) -> (cx, cy, w, h)，均为归一化坐标。"""
    w = box[..., 2] - box[..., 0]
    h = box[..., 3] - box[..., 1]
    cx = (box[..., 0] + box[..., 2]) / 2
    cy = (box[..., 1] + box[..., 3]) / 2
    return torch.stack([cx, cy, w, h], dim=-1)


def jaccard_overlap_np(boxes1: "np.ndarray", boxes2: "np.ndarray") -> "np.ndarray":
    """numpy 版两两 IoU（供评估阶段使用，语义与 :func:`jaccard_overlap` 完全一致）。"""
    import numpy as np

    boxes1 = np.asarray(boxes1, dtype=np.float64).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float64).reshape(-1, 4)
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float64)

    b1 = boxes1[:, None, :]
    b2 = boxes2[None, :, :]
    xmin = np.maximum(b1[..., 0], b2[..., 0])
    ymin = np.maximum(b1[..., 1], b2[..., 1])
    xmax = np.minimum(b1[..., 2], b2[..., 2])
    ymax = np.minimum(b1[..., 3], b2[..., 3])
    iw = np.clip(xmax - xmin, 0, None)
    ih = np.clip(ymax - ymin, 0, None)
    inter = iw * ih

    a1 = np.clip(b1[..., 2] - b1[..., 0], 0, None) * np.clip(b1[..., 3] - b1[..., 1], 0, None)
    a2 = np.clip(b2[..., 2] - b2[..., 0], 0, None) * np.clip(b2[..., 3] - b2[..., 1], 0, None)
    union = a1 + a2 - inter

    iou = np.where(inter > 0, inter / np.maximum(union, 1e-12), 0.0)
    return iou


def scale_bbox(box: torch.Tensor, height: float, width: float) -> torch.Tensor:
    """归一化框 -> 像素框。对应 Caffe ``ScaleBBox``。"""
    out = box.clone()
    out[..., 0] = out[..., 0] * width
    out[..., 1] = out[..., 1] * height
    out[..., 2] = out[..., 2] * width
    out[..., 3] = out[..., 3] * height
    return out
