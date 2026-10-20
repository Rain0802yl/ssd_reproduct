"""先验框（Default Box / Prior Box）生成。

**逐行对齐官方 Caffe 实现** ``src/caffe/layers/prior_box_layer.cpp``：

    * ``aspect_ratios_`` 的构造（第 20-38 行）: 先放入 1.0，再依次放入给定长宽比
      及其**倒数**（``flip=True``）。
    * ``num_priors_`` 的计算（第 39-48 行）: ``len(aspect_ratios_) * len(min_sizes)``，
      若给了 ``max_size`` 则每个再加 1。
    * 框的生成顺序（第 143-195 行）: 先按行 h、再按列 w 遍历特征图位置；
      每个位置内依次输出
          ① min_size 正方形
          ② sqrt(min_size * max_size) 正方形（若有 max_size）
          ③ 其余各长宽比对应的框（跳过 1.0）
      每个长宽比 ar 生成 ``w = min_size*sqrt(ar), h = min_size/sqrt(ar)``。
    * 中心点坐标: ``(index + offset) * step``，``offset = 0.5``。
    * 归一化: 除以 ``img_width`` / ``img_height``。
    * ``clip`` 为 False 时不裁剪。

对 SSD300 + VOC 配置，生成 **8732** 个先验框，与论文完全一致。
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn as nn


def make_aspect_ratios(aspect_ratios: Sequence[float], flip: bool = True) -> List[float]:
    """复刻 Caffe ``PriorBoxLayer::LayerSetUp`` 中长宽比列表的构造过程。

    先放入 1.0；对每个给定长宽比 ar，若尚未存在则放入 ar，
    并在 ``flip`` 为真时同时放入 1/ar。
    """
    out: List[float] = [1.0]
    for ar in aspect_ratios:
        already = any(abs(ar - v) < 1e-6 for v in out)
        if not already:
            out.append(float(ar))
            if flip:
                out.append(1.0 / float(ar))
    return out


class PriorBox(nn.Module):
    """按官方 Caffe 规则生成先验框。

    参数:
        min_dim:      输入图像边长（SSD300 为 300）
        feature_maps: 参与预测的 6 个特征图边长列表，如 [38, 19, 10, 5, 3, 1]
        min_sizes:    每层的 min_size
        max_sizes:    每层的 max_size
        aspect_ratios: 每层给定的长宽比（不含 1 和倒数）
        steps:        每层的步长
        variances:    先验框方差，官方为 [0.1, 0.1, 0.2, 0.2]
        clip:         是否把框裁剪到 [0, 1]，官方配置为 False
        flip:         是否加入长宽比的倒数，官方配置为 True
    """

    def __init__(
        self,
        min_dim: int = 300,
        feature_maps: Sequence[int] = (38, 19, 10, 5, 3, 1),
        min_sizes: Sequence[float] = (30.0, 60.0, 111.0, 162.0, 213.0, 264.0),
        max_sizes: Sequence[float] = (60.0, 111.0, 162.0, 213.0, 264.0, 315.0),
        aspect_ratios: Sequence[Sequence[float]] = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)),
        steps: Sequence[int] = (8, 16, 32, 64, 100, 300),
        variances: Sequence[float] = (0.1, 0.1, 0.2, 0.2),
        clip: bool = False,
        flip: bool = True,
        offset: float = 0.5,
    ) -> None:
        super().__init__()
        assert len(feature_maps) == len(min_sizes) == len(aspect_ratios) == len(steps)
        assert len(max_sizes) == len(min_sizes)

        self.min_dim = float(min_dim)
        self.feature_maps = list(feature_maps)
        self.min_sizes = [float(v) for v in min_sizes]
        self.max_sizes = [float(v) for v in max_sizes]
        self.steps = [float(v) for v in steps]
        self.clip = clip
        self.flip = flip
        self.offset = offset
        self.variances = torch.tensor(variances, dtype=torch.float32)

        # 每层展开后的长宽比列表（含 1.0 与倒数），以及每层每个位置的框数
        self.expanded_aspect_ratios = [make_aspect_ratios(ars, flip) for ars in aspect_ratios]
        self.num_priors_per_layer = []
        for i, ars in enumerate(self.expanded_aspect_ratios):
            # 与 Caffe 一致：len(aspect_ratios_) * len(min_sizes) + (max_size 的个数)
            n = len(ars) * 1  # 本配置每层只有 1 个 min_size
            if self.max_sizes[i] > 0:
                n += 1
            self.num_priors_per_layer.append(n)

    @property
    def num_priors(self) -> int:
        """先验框总数（SSD300 为 8732）。"""
        return sum(
            fm * fm * n for fm, n in zip(self.feature_maps, self.num_priors_per_layer)
        )

    def forward(self) -> torch.Tensor:
        """返回 (num_priors, 4) 的归一化角点坐标先验框。"""
        priors: List[List[float]] = []

        for layer_idx, fm_size in enumerate(self.feature_maps):
            step = self.steps[layer_idx]
            min_size = self.min_sizes[layer_idx]
            max_size = self.max_sizes[layer_idx]
            ars = self.expanded_aspect_ratios[layer_idx]

            for h in range(fm_size):
                for w in range(fm_size):
                    center_x = (w + self.offset) * step
                    center_y = (h + self.offset) * step

                    # ① min_size 正方形
                    box_w = box_h = min_size
                    priors.append(self._make(center_x, center_y, box_w, box_h))

                    # ② sqrt(min_size * max_size) 正方形
                    if max_size > 0:
                        box_w = box_h = math.sqrt(min_size * max_size)
                        priors.append(self._make(center_x, center_y, box_w, box_h))

                    # ③ 其余长宽比
                    for ar in ars:
                        if abs(ar - 1.0) < 1e-6:
                            continue
                        priors.append(
                            self._make(
                                center_x, center_y, min_size * math.sqrt(ar), min_size / math.sqrt(ar)
                            )
                        )

        out = torch.tensor(priors, dtype=torch.float32)
        assert out.shape[0] == self.num_priors, (
            f"先验框数量不一致: 实际 {out.shape[0]}, 预期 {self.num_priors}"
        )
        if self.clip:
            out = out.clamp(0.0, 1.0)
        return out

    def _make(self, cx: float, cy: float, bw: float, bh: float) -> List[float]:
        """由中心点与宽高生成归一化角点坐标。"""
        return [
            (cx - bw / 2.0) / self.min_dim,
            (cy - bh / 2.0) / self.min_dim,
            (cx + bw / 2.0) / self.min_dim,
            (cy + bh / 2.0) / self.min_dim,
        ]


def build_ssd300_prior_box(min_dim: int = 300) -> PriorBox:
    """按官方 ssd_pascal.py 的配置构造 SSD300 先验框生成器。

    官方参数（examples/ssd/ssd_pascal.py 第 297-328 行）::

        min_ratio = 20, max_ratio = 90, step = floor((90-20)/(6-2)) = 17
        min_sizes = [30, 60, 111, 162, 213, 264]
        max_sizes = [60, 111, 162, 213, 264, 315]
        aspect_ratios = [[2], [2,3], [2,3], [2,3], [2], [2]]
        steps = [8, 16, 32, 64, 100, 300]
        prior_variance = [0.1, 0.1, 0.2, 0.2]
    """
    return PriorBox(
        min_dim=min_dim,
        feature_maps=(38, 19, 10, 5, 3, 1),
        min_sizes=(30.0, 60.0, 111.0, 162.0, 213.0, 264.0),
        max_sizes=(60.0, 111.0, 162.0, 213.0, 264.0, 315.0),
        aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)),
        steps=(8, 16, 32, 64, 100, 300),
        variances=(0.1, 0.1, 0.2, 0.2),
        clip=False,
        flip=True,
    )


if __name__ == "__main__":
    pb = build_ssd300_prior_box()
    boxes = pb()
    print("每层每个位置的框数:", pb.num_priors_per_layer)
    print("每层框数:", [fm * fm * n for fm, n in zip(pb.feature_maps, pb.num_priors_per_layer)])
    print("先验框总数:", boxes.shape[0])
    print("前 4 个先验框:\n", boxes[:4])


# --------------------------------------------------------------------------------------
# 配置驱动的构建入口（保证网络头与先验框始终一致）
# --------------------------------------------------------------------------------------
#: 6 个预测层对应的特征图边长（SSD300）
FEATURE_MAP_SIZES: Tuple[int, ...] = (38, 19, 10, 5, 3, 1)


def compute_num_priors_per_layer(
    aspect_ratios: Sequence[Sequence[float]],
    max_sizes: Sequence[float],
    flip: bool = True,
) -> Tuple[int, ...]:
    """计算每一层「每个空间位置」的先验框数量。

    复刻 Caffe ``PriorBoxLayer::LayerSetUp`` 第 39-48 行::

        num_priors_ = len(aspect_ratios_) * len(min_sizes)   # 本配置每层只有 1 个 min_size
        if max_size > 0: num_priors_ += 1

    其中 ``aspect_ratios_`` 已包含 1.0 与各长宽比的倒数（``flip`` 为真时）。

    这个函数必须与 :class:`PriorBox` 使用**同一套参数**，
    否则网络预测头的输出通道数与先验框数量会不匹配。
    """
    out = []
    for ars, mx in zip(aspect_ratios, max_sizes):
        n = len(make_aspect_ratios(ars, flip))
        if mx > 0:
            n += 1
        out.append(n)
    return tuple(out)


def build_prior_box(cfg: Dict) -> "PriorBox":
    """根据配置构建 :class:`PriorBox`。

    支持 ``model.prediction_layers`` 指定只使用部分预测层（用于多尺度消融实验）。
    未被选中的层不会贡献先验框，网络也只在被选中的层上建预测头。
    """
    m = cfg["model"]
    layers = list(m.get("prediction_layers", [0, 1, 2, 3, 4, 5]))
    flip = bool(m.get("flip_priors", True))
    return PriorBox(
        min_dim=m["input_size"],
        feature_maps=[FEATURE_MAP_SIZES[i] for i in layers],
        min_sizes=[m["min_sizes"][i] for i in layers],
        max_sizes=[m["max_sizes"][i] for i in layers],
        aspect_ratios=[m["aspect_ratios"][i] for i in layers],
        steps=[m["steps"][i] for i in layers],
        variances=m["variances"],
        clip=m["clip_prior"],
        flip=flip,
    )


def model_head_spec(cfg: Dict) -> Dict:
    """返回构建网络预测头所需的参数，保证与 :func:`build_prior_box` 一致。"""
    m = cfg["model"]
    layers = list(m.get("prediction_layers", [0, 1, 2, 3, 4, 5]))
    flip = bool(m.get("flip_priors", True))
    return {
        "prediction_layers": tuple(layers),
        "num_priors_per_layer": compute_num_priors_per_layer(
            [m["aspect_ratios"][i] for i in layers],
            [m["max_sizes"][i] for i in layers],
            flip=flip,
        ),
    }
