"""SSD 数据增强 —— 逐环节对齐官方 Caffe 实现。

官方训练时的增强流水线（``src/caffe/layers/annotated_data_layer.cpp`` 第 150-249 行）
严格按以下顺序执行::

    ① DistortImage   光度扰动    -> data_transformer.cpp : ApplyDistort
    ② ExpandImage    随机扩展    -> data_transformer.cpp : ExpandImage(cv::Mat)
    ③ GenerateBatchSamples + CropImage  随机裁剪
                                  -> sampler.cpp : GenerateSamples / SampleBBox
                                  -> data_transformer.cpp : CropImage / TransformAnnotation
    ④ Resize(WARP)   拉伸到 300x300
    ⑤ Mirror         随机水平翻转
    ⑥ 减均值 [104, 117, 123]（BGR 顺序）

各环节的官方参数（``examples/ssd/ssd_pascal.py`` 第 89-221 行）:

    光度扰动: brightness_prob=0.5, brightness_delta=32
              contrast_prob=0.5,   contrast ∈ [0.5, 1.5]
              saturation_prob=0.5, saturation ∈ [0.5, 1.5]
              hue_prob=0.5,        hue_delta=18
              random_order_prob=0.0（不做通道重排）
    随机扩展: prob=0.5, max_expand_ratio=4.0，扩展区域用**均值**填充
    随机裁剪: 7 个采样器（原图 1 个 + IoU 约束 5 个 + 随机裁剪 1 个）
    镜像    : mirror=True

关键细节（易错点）:
    * 光度扰动有**两种随机顺序**（ApplyDistort 中以 0.5 概率二选一）：
          分支 A: 亮度 -> 对比度 -> 饱和度 -> 色相
          分支 B: 亮度 -> 饱和度 -> 色相 -> 对比度
    * 裁剪后只保留**中心点落在裁剪框内**的真值框（emit_constraint=CENTER），
      投影后的框**不裁剪到 [0,1]**（ExtrapolateBBox 对 WARP 模式是空操作）。
    * 扩展区域填充的是**通道均值**，而不是 0 —— 这样减均值后恰好为 0。
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------------------
def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def _jaccard(box1: np.ndarray, box2: np.ndarray) -> float:
    """单个框对单个框的 IoU，对应 Caffe ``JaccardOverlap``。"""
    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    inter = (xmax - xmin) * (ymax - ymin)
    if inter == 0:
        return 0.0
    area1 = max(box1[2] - box1[0], 0) * max(box1[3] - box1[1], 0)
    area2 = max(box2[2] - box2[0], 0) * max(box2[3] - box2[1], 0)
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


# --------------------------------------------------------------------------------------
# ① 光度扰动（对应 ApplyDistort 及其四个子函数）
# --------------------------------------------------------------------------------------
def _adjust_brightness(img: np.ndarray, delta: float) -> np.ndarray:
    """对应 AdjustBrightness：out = in + delta（8 位饱和）。"""
    return np.clip(np.rint(img.astype(np.float32) + delta), 0, 255).astype(np.uint8)


def _adjust_contrast(img: np.ndarray, delta: float) -> np.ndarray:
    """对应 AdjustContrast：out = delta * in（8 位饱和）。"""
    return np.clip(np.rint(img.astype(np.float32) * delta), 0, 255).astype(np.uint8)


def _adjust_saturation(img: np.ndarray, delta: float) -> np.ndarray:
    """对应 AdjustSaturation：转到 HSV，把 S 通道乘以 delta，再转回 BGR。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(np.rint(s.astype(np.float32) * delta), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def _adjust_hue(img: np.ndarray, delta: float) -> np.ndarray:
    """对应 AdjustHue：转到 HSV，把 H 通道加上 delta，再转回 BGR。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = np.clip(np.rint(h.astype(np.float32) + delta), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def apply_distort(
    img: np.ndarray,
    rng: np.random.Generator,
    brightness_prob: float = 0.5,
    brightness_delta: float = 32.0,
    contrast_prob: float = 0.5,
    contrast_lower: float = 0.5,
    contrast_upper: float = 1.5,
    hue_prob: float = 0.5,
    hue_delta: float = 18.0,
    saturation_prob: float = 0.5,
    saturation_lower: float = 0.5,
    saturation_upper: float = 1.5,
) -> np.ndarray:
    """对应 ``ApplyDistort``。注意两种随机顺序。"""

    def maybe(prob: float) -> bool:
        return rng.random() < prob

    # 官方实现：先抽一个 prob，prob > 0.5 走分支 A，否则走分支 B
    branch_a = rng.random() > 0.5

    def do_brightness(im):
        if maybe(brightness_prob):
            d = rng.uniform(-brightness_delta, brightness_delta)
            return _adjust_brightness(im, d)
        return im

    def do_contrast(im):
        if maybe(contrast_prob):
            d = rng.uniform(contrast_lower, contrast_upper)
            return _adjust_contrast(im, d)
        return im

    def do_saturation(im):
        if maybe(saturation_prob):
            d = rng.uniform(saturation_lower, saturation_upper)
            return _adjust_saturation(im, d)
        return im

    def do_hue(im):
        if maybe(hue_prob):
            d = rng.uniform(-hue_delta, hue_delta)
            return _adjust_hue(im, d)
        return im

    if branch_a:
        img = do_brightness(img)
        img = do_contrast(img)
        img = do_saturation(img)
        img = do_hue(img)
    else:
        img = do_brightness(img)
        img = do_saturation(img)
        img = do_hue(img)
        img = do_contrast(img)
    # random_order_prob = 0.0，官方配置下不做通道重排
    return img


# --------------------------------------------------------------------------------------
# ② 随机扩展（对应 ExpandImage(cv::Mat)）
# --------------------------------------------------------------------------------------
def apply_expand(
    img: np.ndarray,
    boxes: np.ndarray,
    mean_bgr: Sequence[float],
    rng: np.random.Generator,
    max_expand_ratio: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """随机把图像放到更大的画布上，扩展区域用通道均值填充。

    返回新的图像与变换后的框（仍在 [0,1] 归一化坐标下）。
    对应 ``DataTransformer::ExpandImage(const cv::Mat&, ...)``。
    """
    h, w = img.shape[:2]
    ratio = rng.uniform(1.0, max_expand_ratio)
    new_h = int(h * ratio)
    new_w = int(w * ratio)
    if new_h <= h or new_w <= w:
        return img, boxes

    h_off = int(math.floor(rng.uniform(0.0, float(new_h - h))))
    w_off = int(math.floor(rng.uniform(0.0, float(new_w - w))))

    # 用均值填充（对应官方先 setTo(0) 再用 mean_values 覆盖）
    canvas = np.empty((new_h, new_w, 3), dtype=np.uint8)
    canvas[:, :] = np.array(mean_bgr, dtype=np.uint8)
    canvas[h_off : h_off + h, w_off : w_off + w] = img

    if boxes.size > 0:
        # 原图在画布中的归一化位置
        xmin = -w_off / w
        ymin = -h_off / h
        xmax = (new_w - w_off) / w
        ymax = (new_h - h_off) / h
        scaled = boxes.copy()
        scaled[:, 0] = xmin + boxes[:, 0] * (xmax - xmin)
        scaled[:, 1] = ymin + boxes[:, 1] * (ymax - ymin)
        scaled[:, 2] = xmin + boxes[:, 2] * (xmax - xmin)
        scaled[:, 3] = ymin + boxes[:, 3] * (ymax - ymin)
    else:
        scaled = boxes

    return canvas, scaled


# --------------------------------------------------------------------------------------
# ③ 随机裁剪（对应 sampler.cpp + CropImage/TransformAnnotation）
# --------------------------------------------------------------------------------------
class BatchSampler:
    """对应 proto 消息 ``BatchSampler``。"""

    def __init__(
        self,
        min_scale: float = 1.0,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 1.0,
        max_aspect_ratio: float = 1.0,
        min_jaccard_overlap: Optional[float] = None,
        max_jaccard_overlap: Optional[float] = None,
        max_trials: int = 100,
        max_sample: int = 1,
    ) -> None:
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_jaccard_overlap = min_jaccard_overlap
        self.max_jaccard_overlap = max_jaccard_overlap
        self.max_trials = max_trials
        self.max_sample = max_sample

    # ---- 对应 SampleBBox ----
    def sample_bbox(self, rng: np.random.Generator) -> np.ndarray:
        scale = rng.uniform(self.min_scale, self.max_scale)
        aspect_ratio = rng.uniform(self.min_aspect_ratio, self.max_aspect_ratio)
        # 官方对长宽比做了与 scale 相关的夹取
        aspect_ratio = max(aspect_ratio, scale ** 2)
        aspect_ratio = min(aspect_ratio, 1.0 / (scale ** 2))

        bbox_w = scale * math.sqrt(aspect_ratio)
        bbox_h = scale / math.sqrt(aspect_ratio)

        w_off = rng.uniform(0.0, 1.0 - bbox_w) if bbox_w < 1.0 else 0.0
        h_off = rng.uniform(0.0, 1.0 - bbox_h) if bbox_h < 1.0 else 0.0
        return np.array([w_off, h_off, w_off + bbox_w, h_off + bbox_h], dtype=np.float64)

    # ---- 对应 SatisfySampleConstraint ----
    def satisfy(self, sampled: np.ndarray, gt_boxes: np.ndarray) -> bool:
        has_jaccard = (
            self.min_jaccard_overlap is not None or self.max_jaccard_overlap is not None
        )
        if not has_jaccard:
            # 没有任何约束时，官方直接返回 true
            return True
        for gt in gt_boxes:
            overlap = _jaccard(sampled, gt)
            if self.min_jaccard_overlap is not None and overlap < self.min_jaccard_overlap:
                continue
            if self.max_jaccard_overlap is not None and overlap > self.max_jaccard_overlap:
                continue
            return True
        return False

    # ---- 对应 GenerateSamples ----
    def generate(self, gt_boxes: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
        found: List[np.ndarray] = []
        for _ in range(self.max_trials):
            if self.max_sample and len(found) >= self.max_sample:
                break
            sampled = self.sample_bbox(rng)
            if self.satisfy(sampled, gt_boxes):
                found.append(sampled)
        return found


def default_batch_samplers() -> List[BatchSampler]:
    """官方 ``ssd_pascal.py`` 中的 7 个 batch_sampler（第 89-174 行）。"""
    scale_kw = dict(min_scale=0.3, max_scale=1.0, min_aspect_ratio=0.5, max_aspect_ratio=2.0)
    return [
        # ① 原图（sampler 全部取默认值 -> 单位框）
        BatchSampler(max_trials=1, max_sample=1),
        # ②~⑥ 逐步放宽的 IoU 下限
        BatchSampler(min_jaccard_overlap=0.1, max_trials=50, max_sample=1, **scale_kw),
        BatchSampler(min_jaccard_overlap=0.3, max_trials=50, max_sample=1, **scale_kw),
        BatchSampler(min_jaccard_overlap=0.5, max_trials=50, max_sample=1, **scale_kw),
        BatchSampler(min_jaccard_overlap=0.7, max_trials=50, max_sample=1, **scale_kw),
        BatchSampler(min_jaccard_overlap=0.9, max_trials=50, max_sample=1, **scale_kw),
        # ⑦ 纯随机裁剪
        BatchSampler(max_jaccard_overlap=1.0, max_trials=50, max_sample=1, **scale_kw),
    ]


def apply_crop(
    img: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    crop: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 ``crop``（归一化角点）裁剪图像并变换标注。

    对应 ``DataTransformer::CropImage(const AnnotatedDatum&, ...)`` +
    ``TransformAnnotation``：
      * 只保留**中心点在裁剪框内**的框（emit_constraint = CENTER）
      * 与裁剪框完全不相交的框丢弃（ProjectBBox 返回 false）
      * 投影后的框**不裁剪到 [0,1]**
    """
    h, w = img.shape[:2]
    crop = np.clip(crop, 0.0, 1.0)

    xmin = int(crop[0] * w)
    ymin = int(crop[1] * h)
    xmax = int(crop[2] * w)
    ymax = int(crop[3] * h)
    xmax = max(xmax, xmin + 1)
    ymax = max(ymax, ymin + 1)
    xmax = min(xmax, w)
    ymax = min(ymax, h)

    cropped = img[ymin:ymax, xmin:xmax].copy()

    if boxes.size == 0:
        return cropped, boxes, labels

    cw = crop[2] - crop[0]
    ch = crop[3] - crop[1]
    keep_boxes: List[np.ndarray] = []
    keep_labels: List[int] = []
    for i in range(boxes.shape[0]):
        box = boxes[i]
        # ---- emit_constraint = CENTER ----
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        if not (crop[0] <= cx <= crop[2] and crop[1] <= cy <= crop[3]):
            continue
        # ---- ProjectBBox：完全不相交则丢弃 ----
        if box[0] >= crop[2] or box[2] <= crop[0] or box[1] >= crop[3] or box[3] <= crop[1]:
            continue
        proj = np.array(
            [
                (box[0] - crop[0]) / cw,
                (box[1] - crop[1]) / ch,
                (box[2] - crop[0]) / cw,
                (box[3] - crop[1]) / ch,
            ],
            dtype=np.float64,
        )
        # 注意：官方在此处不裁剪到 [0,1]
        keep_boxes.append(proj)
        keep_labels.append(int(labels[i]))

    if not keep_boxes:
        return cropped, np.zeros((0, 4), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    return (
        cropped,
        np.stack(keep_boxes).astype(np.float64),
        np.array(keep_labels, dtype=np.int64),
    )


# --------------------------------------------------------------------------------------
# 组合成完整的训练/测试变换
# --------------------------------------------------------------------------------------
class SSDTransform:
    """SSD 的输入变换。

    参数:
        resize:            输出边长（SSD300 为 300）
        mean_bgr:          通道均值，官方为 [104, 117, 123]（BGR 顺序）
        train:             是否为训练模式（训练才做增强）
        use_distort/use_expand/use_crop/use_mirror: 消融实验开关
    """

    def __init__(
        self,
        resize: int = 300,
        mean_bgr: Sequence[float] = (104.0, 117.0, 123.0),
        train: bool = True,
        use_distort: bool = True,
        use_expand: bool = True,
        use_crop: bool = True,
        use_mirror: bool = True,
        expand_prob: float = 0.5,
        max_expand_ratio: float = 4.0,
        batch_samplers: Optional[List[BatchSampler]] = None,
        seed: Optional[int] = None,
        norm_mode: str = "imagenet",
    ) -> None:
        self.resize = resize
        self.mean_bgr = np.array(mean_bgr, dtype=np.float32)
        self.train = train
        self.use_distort = use_distort
        self.use_expand = use_expand
        self.use_crop = use_crop
        self.use_mirror = use_mirror
        self.expand_prob = expand_prob
        self.max_expand_ratio = max_expand_ratio
        self.batch_samplers = batch_samplers if batch_samplers is not None else default_batch_samplers()
        self.rng = np.random.default_rng(seed)
        # 输入归一化方式：
        #   "imagenet" —— (x/255 - ImageNet均值) / ImageNet标准差，范围约 ±2.6
        #                 与 torchvision 的 VGG16 预训练权重匹配（**推荐**）
        #   "caffe"    —— BGR 直接减 [104,117,123]，范围约 ±132
        #                 官方 Caffe SSD 的约定，但那是配作者自己的 Caffe 权重
        assert norm_mode in ("imagenet", "caffe"), "norm_mode 只能是 'imagenet' 或 'caffe'"
        self.norm_mode = norm_mode
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(
        self, img: np.ndarray, boxes: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        参数:
            img:    HWC uint8 BGR
            boxes:  (G, 4) 归一化角点坐标
            labels: (G,)
        返回:
            (CHW float32 已减均值的图像, 变换后的框, 标签)
        """
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)

        if self.train:
            # ① 光度扰动
            if self.use_distort:
                img = apply_distort(img, self.rng)

            # ② 随机扩展
            if self.use_expand and self.rng.random() < self.expand_prob:
                img, boxes = apply_expand(
                    img, boxes, self.mean_bgr, self.rng, self.max_expand_ratio
                )

            # ③ 随机裁剪
            if self.use_crop and boxes.size > 0:
                candidates: List[np.ndarray] = []
                for sampler in self.batch_samplers:
                    candidates.extend(sampler.generate(boxes, self.rng))
                if candidates:
                    pick = candidates[self.rng.integers(len(candidates))]
                    img, boxes, labels = apply_crop(img, boxes, labels, pick)

            # ⑤ 随机水平翻转（在 resize 之前，与官方 Transform 的顺序一致）
            if self.use_mirror and self.rng.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                if boxes.size > 0:
                    xmin = boxes[:, 0].copy()
                    boxes[:, 0] = 1.0 - boxes[:, 2]
                    boxes[:, 2] = 1.0 - xmin

        # ④ 缩放（WARP：直接拉伸，不保持长宽比）
        img = cv2.resize(img, (self.resize, self.resize), interpolation=cv2.INTER_LINEAR)

        # ⑥ 归一化 + 转 CHW
        if self.norm_mode == "imagenet":
            # BGR -> RGB，再 /255 并按 ImageNet 均值方差归一化
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - self.imagenet_mean.reshape(1, 1, 3)) / self.imagenet_std.reshape(1, 1, 3)
        else:
            img = img.astype(np.float32) - self.mean_bgr.reshape(1, 1, 3)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))

        return img, boxes.astype(np.float32), labels
