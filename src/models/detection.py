"""推理后处理：框解码 + 逐类 NMS + top-k 筛选。

**对齐官方 Caffe 实现** ``src/caffe/layers/detection_output_layer.cpp``：

    解码              ->  GetLocPredictions + DecodeBBoxesAll（CENTER_SIZE）
    clip_bbox = false ->  第 205 行 ``const bool clip_bbox = false;``，解码后**不裁剪**
    逐类 NMS          ->  第 235 行 ApplyNMSFast(score_threshold=0.01, nms_threshold=0.45, top_k=400)
    保留 top-k        ->  第 239-270 行，keep_top_k = 200（跨类别按得分取前 200）

官方 det_out_param（examples/ssd/ssd_pascal.py 第 391-408 行）::

    nms_threshold = 0.45, top_k = 400
    keep_top_k = 200
    confidence_threshold = 0.01
    background_label_id = 0
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from ..utils.box_utils import decode_bbox


@torch.no_grad()
def decode_predictions(
    loc_pred: torch.Tensor,
    priors: torch.Tensor,
    variances: torch.Tensor,
) -> torch.Tensor:
    """把定位预测解码成归一化角点坐标框。

    参数:
        loc_pred: (N, P, 4)
        priors:   (P, 4)
        variances:(4,)
    返回:
        (N, P, 4) 解码后的框（**不裁剪**，与官方 clip_bbox=false 一致）
    """
    return decode_bbox(priors.unsqueeze(0), loc_pred, variances.view(1, 1, 4))


@torch.no_grad()
def nms_per_class(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    num_classes: int,
    background_label_id: int = 0,
    confidence_threshold: float = 0.01,
    nms_threshold: float = 0.45,
    top_k: int = 400,
    keep_top_k: int = 200,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """对单张图执行逐类 NMS 与跨类别 top-k 保留。

    参数:
        boxes:  (P, 4) 解码后的框
        scores: (P, C) softmax 后的类别概率
    返回:
        列表，每项为 ``(类别下标, 得分, 框)``；按得分从高到低排序，最多 ``keep_top_k`` 项。
    """
    from torchvision.ops import nms  # 延迟导入，便于单测

    keep_cls: List[torch.Tensor] = []
    keep_score: List[torch.Tensor] = []
    keep_box: List[torch.Tensor] = []

    for c in range(num_classes):
        if c == background_label_id:
            continue
        cls_scores = scores[:, c]
        # 与 Caffe GetMaxScoreIndex 一致：严格大于阈值
        cand = (cls_scores > confidence_threshold).nonzero(as_tuple=True)[0]
        if cand.numel() == 0:
            continue

        cand_scores = cls_scores[cand]
        # 先按得分取前 top_k 个（Caffe ApplyNMSFast 会先裁剪到 top_k）
        if top_k > -1 and cand.numel() > top_k:
            order = torch.topk(cand_scores, top_k).indices
            cand = cand[order]
            cand_scores = cand_scores[order]

        keep = nms(boxes[cand], cand_scores, nms_threshold)
        kept = cand[keep]
        keep_cls.append(torch.full((kept.numel(),), c, dtype=torch.long, device=boxes.device))
        keep_score.append(cls_scores[kept])
        keep_box.append(boxes[kept])

    if not keep_score:
        empty = boxes.new_zeros((0, 4))
        return [
            (
                torch.zeros(0, dtype=torch.long, device=boxes.device),
                boxes.new_zeros((0,)),
                empty,
            )
        ]

    cls_all = torch.cat(keep_cls)
    score_all = torch.cat(keep_score)
    box_all = torch.cat(keep_box)

    # 跨类别保留 keep_top_k（对应 Caffe 第 239-270 行）
    if keep_top_k > -1 and score_all.numel() > keep_top_k:
        order = torch.topk(score_all, keep_top_k).indices
        cls_all, score_all, box_all = cls_all[order], score_all[order], box_all[order]

    return [(cls_all, score_all, box_all)]


@torch.no_grad()
def detect_batch(
    loc_pred: torch.Tensor,
    conf_pred: torch.Tensor,
    priors: torch.Tensor,
    variances: torch.Tensor,
    num_classes: int = 21,
    background_label_id: int = 0,
    confidence_threshold: float = 0.01,
    nms_threshold: float = 0.45,
    top_k: int = 400,
    keep_top_k: int = 200,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """批量检测：解码 -> softmax -> 逐类 NMS。

    返回长度为 N 的列表，每项为 ``(类别下标, 得分, 框)``。
    """
    probs = torch.softmax(conf_pred, dim=-1)
    decoded = decode_predictions(loc_pred, priors, variances)
    results = []
    for b in range(decoded.shape[0]):
        results.extend(
            nms_per_class(
                decoded[b],
                probs[b],
                num_classes=num_classes,
                background_label_id=background_label_id,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                top_k=top_k,
                keep_top_k=keep_top_k,
            )
        )
    return results
