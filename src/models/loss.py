"""MultiBoxLoss：SSD 的多框损失（含难负样本挖掘）。

**逐段对齐官方 Caffe 实现**：

    匹配策略   ->  src/caffe/util/bbox_util.cpp : MatchBBox      （第 584-720 行）
    难负样本   ->  src/caffe/util/bbox_util.cpp : MineHardExamples（第 849-1030 行）
    损失合成   ->  src/caffe/layers/multibox_loss_layer.cpp      （第 241-252 行）

损失公式（论文式 1）::

    L = (1 / N) * ( L_conf + alpha * L_loc )

其中 N 为本 batch 中匹配上的正样本总数，alpha = loc_weight = 1.0。

匹配规则（两阶段，与 Caffe 完全一致）：
    阶段一（BIPARTITE 贪心）: 反复找出全局 IoU 最大的「未匹配先验框 × 未占用真值框」，
        进行配对。这一步保证**每个真值框至少有一个先验框负责**。
    阶段二（PER_PREDICTION）: 对剩余未匹配的先验框，若与任一真值框的 IoU >= 0.5
        （``overlap_threshold``），则与其中 IoU 最大的真值框匹配。

难负样本挖掘（``MINING_TYPE = MAX_NEGATIVE``）：
    负样本需满足 ``未匹配`` 且 ``与所有真值框的最大 IoU < 0.5``（``neg_overlap``）；
    按置信度损失从大到小排序，取前 ``min(正样本数 * 3, 合格负样本数)`` 个。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.box_utils import encode_bbox, jaccard_overlap


def match_priors_to_gt(
    priors: torch.Tensor,
    gt_boxes: torch.Tensor,
    overlap_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """复刻 Caffe ``MatchBBox`` 的两阶段匹配。

    参数:
        priors: (P, 4) 归一化角点坐标
        gt_boxes: (G, 4) 归一化角点坐标
    返回:
        match_idx:     (P,)  每个先验框匹配到的真值框下标，未匹配为 -1
        match_overlap: (P,)  每个先验框与所有真值框的最大 IoU（无重叠则为 0）
    """
    num_priors = priors.shape[0]
    num_gt = gt_boxes.shape[0]
    device = priors.device

    iou = jaccard_overlap(priors, gt_boxes)  # (P, G)

    match_idx = torch.full((num_priors,), -1, dtype=torch.long, device=device)
    max_iou = iou.max(dim=1).values if num_gt > 0 else torch.zeros(num_priors, device=device)
    # Caffe 只记录 overlap > 1e-6 的配对，其余保持 0（结果与最大 IoU 一致）
    match_overlap = torch.where(max_iou > 1e-6, max_iou, torch.zeros_like(max_iou))

    if num_gt == 0 or num_priors == 0:
        return match_idx, match_overlap

    # ---- 阶段一：二部图贪心匹配 ----
    # 只在有重叠的候选上做，避免在 8732 x G 的全矩阵上反复取最大值
    has_overlap = (iou > 1e-6).any(dim=1)
    cand = has_overlap.nonzero(as_tuple=True)[0]
    if cand.numel() > 0:
        sub = iou[cand]  # (K, G)
        free_prior = torch.ones(cand.numel(), dtype=torch.bool, device=device)
        free_gt = torch.ones(num_gt, dtype=torch.bool, device=device)
        for _ in range(num_gt):
            m = sub.clone()
            m[~free_prior] = -1.0
            m[:, ~free_gt] = -1.0
            flat = int(m.argmax())
            k, j = divmod(flat, num_gt)
            if float(m[k, j]) <= 1e-6:
                break
            match_idx[cand[k]] = j
            match_overlap[cand[k]] = sub[k, j]
            free_prior[k] = False
            free_gt[j] = False

    # ---- 阶段二：PER_PREDICTION ----
    unmatched = match_idx == -1
    if unmatched.any():
        sub = iou[unmatched]  # (M, G)
        best, best_j = sub.max(dim=1)
        ok = best >= overlap_threshold
        if ok.any():
            idx = unmatched.nonzero(as_tuple=True)[0]
            match_idx[idx[ok]] = best_j[ok]
            match_overlap[idx[ok]] = best[ok]

    return match_idx, match_overlap


class MultiBoxLoss(nn.Module):
    """SSD 多框损失。

    参数（默认值即官方 ssd_pascal.py 的配置）:
        priors:             (P, 4) 先验框
        variances:          (4,)   [0.1, 0.1, 0.2, 0.2]
        num_classes:        21
        neg_pos_ratio:      3.0    正负样本比例
        overlap_threshold:  0.5    匹配阈值
        neg_overlap:        0.5    负样本的 IoU 上限
        background_label_id: 0
        loc_weight:         1.0    定位损失权重 alpha
    """

    def __init__(
        self,
        priors: torch.Tensor,
        variances: torch.Tensor = torch.tensor([0.1, 0.1, 0.2, 0.2]),
        num_classes: int = 21,
        neg_pos_ratio: float = 3.0,
        overlap_threshold: float = 0.5,
        neg_overlap: float = 0.5,
        background_label_id: int = 0,
        loc_weight: float = 1.0,
        mining_type: str = "max_negative",
        normalize_conf_by_samples: bool = False,
    ) -> None:
        super().__init__()
        assert mining_type in ("max_negative", "none"), (
            "mining_type 只能是 'max_negative'（官方难负样本挖掘）或 'none'（使用全部合格负样本）"
        )
        self.mining_type = mining_type
        # 官方（VALID 归一化）把定位损失与分类损失**都**除以「匹配上的正样本数 N」。
        # 这在开启难负样本挖掘时量级是合理的。但一旦关闭挖掘，负样本数量会暴增
        # 约 50 倍（≈13 万 vs ≈500），分类损失也随之放大 50 倍，训练直接发散。
        # 为做「只改变挖掘开关」的公平对照，A4 打开此选项：
        # 定位损失除以正样本数，分类损失除以它自己的样本数，量级与基线可比。
        self.normalize_conf_by_samples = bool(normalize_conf_by_samples)
        self.register_buffer("priors", priors.float())
        self.register_buffer("variances", variances.float())
        self.num_classes = num_classes
        self.neg_pos_ratio = float(neg_pos_ratio)
        self.overlap_threshold = float(overlap_threshold)
        self.neg_overlap = float(neg_overlap)
        self.background_label_id = int(background_label_id)
        self.loc_weight = float(loc_weight)

    def forward(
        self,
        loc_pred: torch.Tensor,
        conf_pred: torch.Tensor,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        参数:
            loc_pred:  (N, P, 4)
            conf_pred: (N, P, C)
            gt_boxes:  长度为 N 的列表，每项 (Gi, 4)，归一化角点坐标
            gt_labels: 长度为 N 的列表，每项 (Gi,) LongTensor（1~20，0 保留给背景）
        返回:
            (总损失, 统计信息字典)
        """
        device = loc_pred.device
        priors = self.priors.to(device)
        variances = self.variances.to(device)
        batch_size = loc_pred.shape[0]

        loc_loss_sum = torch.zeros((), device=device)
        conf_loss_sum = torch.zeros((), device=device)
        total_pos = 0
        total_neg = 0
        total_conf_samples = 0

        for b in range(batch_size):
            boxes = gt_boxes[b].to(device).float()
            labels = gt_labels[b].to(device).long()

            if boxes.numel() == 0:
                match_idx = torch.full(
                    (priors.shape[0],), -1, dtype=torch.long, device=device
                )
                match_overlap = torch.zeros(priors.shape[0], device=device)
            else:
                match_idx, match_overlap = match_priors_to_gt(
                    priors, boxes, self.overlap_threshold
                )

            pos_mask = match_idx > -1
            num_pos = int(pos_mask.sum())

            # ---------- 定位损失：只对正样本计算 ----------
            if num_pos > 0:
                target = encode_bbox(
                    priors[pos_mask], boxes[match_idx[pos_mask]], variances
                )
                loc_loss_sum = loc_loss_sum + F.smooth_l1_loss(
                    loc_pred[b][pos_mask], target, reduction="sum", beta=1.0
                )

            # ---------- 分类损失 ----------
            # 正样本的目标类别 = 匹配到的真值框类别；负样本 = 背景
            conf_target = torch.full(
                (priors.shape[0],), self.background_label_id, dtype=torch.long, device=device
            )
            if num_pos > 0:
                conf_target[pos_mask] = labels[match_idx[pos_mask]]

            conf_loss_all = F.cross_entropy(
                conf_pred[b], conf_target, reduction="none"
            )  # (P,)

            # 正样本全都要
            if num_pos > 0:
                conf_loss_sum = conf_loss_sum + conf_loss_all[pos_mask].sum()

            # ---------- 难负样本挖掘 ----------
            # 合格负样本：未匹配 且 与所有真值框的最大 IoU < neg_overlap
            neg_eligible = (~pos_mask) & (match_overlap < self.neg_overlap)
            if self.mining_type == "none":
                # 消融实验：不做难负样本挖掘，所有合格负样本都参与分类损失
                num_neg = int(neg_eligible.sum())
            else:
                # 官方 MAX_NEGATIVE：按置信度损失降序，取 min(正样本数*3, 合格负样本数) 个
                num_neg = min(
                    int(num_pos * self.neg_pos_ratio), int(neg_eligible.sum())
                )
            if num_neg > 0:
                neg_scores = conf_loss_all.masked_fill(~neg_eligible, float("-inf"))
                top_idx = torch.topk(neg_scores, num_neg).indices
                conf_loss_sum = conf_loss_sum + conf_loss_all[top_idx].sum()
                total_neg += num_neg

            total_pos += num_pos
            total_conf_samples += num_pos + num_neg

        # ---------- 归一化 ----------
        # 官方：定位与分类损失都除以整个 batch 的正样本总数（Caffe 的 VALID 归一化）
        normalizer = float(max(total_pos, 1))
        conf_normalizer = (
            float(max(total_conf_samples, 1))
            if self.normalize_conf_by_samples
            else normalizer
        )
        loc_loss = loc_loss_sum * self.loc_weight / normalizer
        conf_loss = conf_loss_sum / conf_normalizer
        total_loss = loc_loss + conf_loss

        stats = {
            "loss": float(total_loss.detach()),
            "loc_loss": float(loc_loss.detach()),
            "conf_loss": float(conf_loss.detach()),
            "num_pos": total_pos,
            "num_neg": total_neg,
            "num_conf_samples": total_conf_samples,
        }
        return total_loss, stats
