"""VOC 检测指标（mAP）。

**对齐官方 Caffe 实现**：

    ``src/caffe/layers/detection_evaluate_layer.cpp``（第 190-233 行）
        评测时的匹配与 TP/FP 判定
    ``src/caffe/util/bbox_util.cpp : ComputeAP``
        11 点插值 AP（``ap_version = "11point"``）

判定规则（``evaluate_difficult_gt = False``，即官方 ssd_pascal.py 的配置）::

    对每个类别，把所有检测框按得分从高到低排序，逐个与同类真值框比较：
      * 与所有真值框的最大 IoU >= 0.5：
          - 若该真值框**不是** difficult：
              - 该真值框此前未被占用 -> 真阳性 (TP)
              - 已被占用              -> 假阳性 (FP)
          - 若该真值框**是** difficult：该检测被**完全忽略**（既不算 TP 也不算 FP）
      * 最大 IoU < 0.5：假阳性 (FP)

AP 采用 VOC2007 的 **11 点插值**：对 recall = 0, 0.1, ..., 1.0 共 11 个点，
取「recall 不低于该点」范围内的最大 precision，最后取平均。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .box_utils import jaccard_overlap_np


def compute_ap_11point(
    tp: np.ndarray, fp: np.ndarray, num_pos: int
) -> Tuple[float, np.ndarray, np.ndarray]:
    """11 点插值 AP。对应 Caffe ``ComputeAP`` 的 ``ap_version == "11point"`` 分支。

    参数:
        tp:       (D,) 每个检测是否为真阳性（0/1），**已按得分降序排列**
        fp:       (D,) 每个检测是否为假阳性（0/1），**已按得分降序排列**
        num_pos:  该类别的真值框数量（不含 difficult）
    返回:
        (ap, precision 数组, recall 数组)
    """
    if tp.size == 0 or num_pos == 0:
        return 0.0, np.zeros(0), np.zeros(0)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    denom = tp_cum + fp_cum
    prec = np.where(denom > 0, tp_cum / np.maximum(denom, 1e-12), 0.0)
    rec = tp_cum / float(num_pos)

    max_precs = np.zeros(11, dtype=np.float64)
    start_idx = len(rec) - 1
    for j in range(10, -1, -1):
        i = start_idx
        while i >= 0:
            if rec[i] < j / 10.0:
                start_idx = i
                if j > 0:
                    max_precs[j - 1] = max_precs[j]
                break
            else:
                if max_precs[j] < prec[i]:
                    max_precs[j] = prec[i]
            i -= 1
    ap = float(max_precs.sum() / 11.0)
    return ap, prec, rec


def evaluate_detections(
    all_detections: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    all_groundtruths: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_classes: int = 21,
    background_label_id: int = 0,
    overlap_threshold: float = 0.5,
    evaluate_difficult_gt: bool = False,
    class_names: Sequence[str] = (),
) -> Dict:
    """计算 VOC 风格的 mAP。

    参数:
        all_detections:    长度 N 的序列，每项 ``(labels (D,), scores (D,), boxes (D,4))``
        all_groundtruths:  长度 N 的序列，每项 ``(boxes (G,4), labels (G,), difficult (G,))``
        num_classes:       类别总数（含背景）
        overlap_threshold: IoU 阈值，官方为 0.5
        evaluate_difficult_gt: 是否把 difficult 真值框计入评测，官方为 False
    返回:
        含 ``mAP``、``ap_per_class``、``pr_curves`` 等字段的字典
    """
    ap_per_class: Dict[int, float] = {}
    pr_curves: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for c in range(num_classes):
        if c == background_label_id:
            continue

        # ---------- 收集该类别的真值框 ----------
        gt_per_image: List[Tuple[np.ndarray, np.ndarray, List[int]]] = []
        num_pos = 0
        for boxes, labels, difficult in all_groundtruths:
            sel = labels == c
            gt_c = boxes[sel]
            diff_c = difficult[sel].astype(bool)
            gt_per_image.append((gt_c, diff_c, []))
            if evaluate_difficult_gt:
                num_pos += int(gt_c.shape[0])
            else:
                num_pos += int((~diff_c).sum())

        # ---------- 收集该类别的检测框 ----------
        dets: List[Tuple[float, int, np.ndarray]] = []
        for img_idx, (labels, scores, boxes) in enumerate(all_detections):
            sel = labels == c
            for sc, bx in zip(scores[sel], boxes[sel]):
                dets.append((float(sc), img_idx, bx))
        dets.sort(key=lambda t: -t[0])

        # ---------- 逐个检测判定 TP / FP ----------
        visited = [np.zeros(gt_per_image[i][0].shape[0], dtype=bool) for i in range(len(gt_per_image))]
        tp_list: List[int] = []
        fp_list: List[int] = []

        for score, img_idx, box in dets:
            gt_c, diff_c, _ = gt_per_image[img_idx]
            if gt_c.shape[0] == 0:
                tp_list.append(0)
                fp_list.append(1)
                continue
            ious = jaccard_overlap_np(box[None, :], gt_c)[0]
            jmax = int(np.argmax(ious))
            overlap_max = float(ious[jmax])

            if overlap_max >= overlap_threshold:
                is_difficult = bool(diff_c[jmax])
                if evaluate_difficult_gt or not is_difficult:
                    if not visited[img_idx][jmax]:
                        tp_list.append(1)
                        fp_list.append(0)
                        visited[img_idx][jmax] = True
                    else:
                        tp_list.append(0)
                        fp_list.append(1)
                # else: 匹配到 difficult 真值框 -> 该检测被完全忽略
            else:
                tp_list.append(0)
                fp_list.append(1)

        tp = np.array(tp_list, dtype=np.float64)
        fp = np.array(fp_list, dtype=np.float64)
        ap, prec, rec = compute_ap_11point(tp, fp, num_pos)
        ap_per_class[c] = ap
        pr_curves[c] = (prec, rec)

    valid_aps = [v for v in ap_per_class.values() if v > 0] if ap_per_class else []
    mAP = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    mAP_nonzero = float(np.mean(valid_aps)) if valid_aps else 0.0

    return {
        "mAP": mAP,
        "mAP_nonzero": mAP_nonzero,
        "ap_per_class": ap_per_class,
        "pr_curves": pr_curves,
        "class_names": list(class_names),
    }
