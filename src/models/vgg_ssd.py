"""SSD300 网络结构（VGG16 主干 + 多尺度预测头）。

**逐层对齐官方 Caffe 实现**：

    主干        ->  python/caffe/model_libs.py : VGGNetBody(fully_conv=True, reduced=True, dilated=True)
    额外层      ->  examples/ssd/ssd_pascal.py : AddExtraLayers
    预测头      ->  python/caffe/model_libs.py : CreateMultiBoxHead
    L2 归一化   ->  CreateMultiBoxHead 中的 L.Normalize(scale=20, across_spatial=False, channel_shared=False)

结构要点（输入 300x300）::

    conv1_1 ~ relu4_3                      -> 38 x 38 x 512   (预测层 1，先做 L2 归一化 scale=20)
    pool4(2x2 s2)                          -> 19 x 19
    conv5_1 ~ relu5_3                      -> 19 x 19 x 512
    pool5(3x3 s1 pad1)                     -> 19 x 19          <- 注意：步长 1，不降采样
    conv6  (3x3, dilation=6, pad=6, 1024)  -> 19 x 19          <- fc6 转空洞卷积
    conv7  (1x1, 1024)                     -> 19 x 19          (预测层 2)
    conv8_1(1x1,256) conv8_2(3x3 s2 p1,512)-> 10 x 10          (预测层 3)
    conv9_1(1x1,128) conv9_2(3x3 s2 p1,256)->  5 x  5          (预测层 4)
    conv10_1(1x1,128) conv10_2(3x3 p0,256) ->  3 x  3          (预测层 5)
    conv11_1(1x1,128) conv11_2(3x3 p0,256) ->  1 x  1          (预测层 6)

预测头：每层接一个 3x3 (pad=1) 卷积，分别输出 ``num_priors*4``（定位）与
``num_priors*num_classes``（分类）个通道，即论文中的 3x3x(4*(C+4)) 与 3x3x(C*21) 卷积核。

主干权重初始化方式与 torchvision 官方 ``ssd300_vgg16`` 一致：
conv1_1~conv5_3 使用 ImageNet 预训练权重；conv6/conv7 与额外层用 Xavier 初始化。
（torchvision 官方实现也是丢弃原 VGG16 的 fc6/fc7 并重新 Xavier 初始化，
 而非把 4096x25088 的全连接权重强行降采样成 3x3 空洞卷积。）
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# 官方 SSD300 配置下的默认值（与 ssd_pascal.py 一致）
NUM_PRIORS_PER_LAYER: Tuple[int, ...] = (4, 6, 6, 6, 4, 4)
FEATURE_MAP_SIZES: Tuple[int, ...] = (38, 19, 10, 5, 3, 1)
ALL_PREDICTION_LAYERS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
#: 6 个预测层各自作为预测头输入时的通道数
HEAD_IN_CHANNELS: Tuple[int, ...] = (512, 1024, 512, 256, 256, 256)


def _xavier_init(module: nn.Module) -> None:
    """对齐 Caffe 的 ``weight_filler: xavier`` / ``bias_filler: constant 0``。"""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class L2Norm(nn.Module):
    """conv4_3 的 L2 归一化 + 逐通道缩放。

    对应 Caffe 的 ``Normalize`` 层（``across_spatial=False``, ``channel_shared=False``）:
    在每个空间位置上对**通道维**求 L2 范数并归一化，然后乘以逐通道的可学习缩放系数。

    官方配置中 ``normalizations = [20, -1, -1, -1, -1, -1]``，即只有 conv4_3 做这个操作，
    初始缩放系数为 20。
    """

    def __init__(self, num_channels: int = 512, scale: float = 20.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((num_channels,), scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight.view(1, -1, 1, 1) * F.normalize(x, p=2, dim=1)


class SSD300(nn.Module):
    """SSD300：VGG16 主干 + 6 个尺度的检测头。

    参数:
        num_classes:   类别数（含背景），VOC 为 21
        pretrained:    是否加载 ImageNet 预训练的主干权重
        l2_norm_scale: conv4_3 的 L2 归一化初始缩放系数，官方为 20
    """

    def __init__(
        self,
        num_classes: int = 21,
        pretrained: bool = True,
        l2_norm_scale: float = 20.0,
        num_priors_per_layer: Tuple[int, ...] = NUM_PRIORS_PER_LAYER,
        prediction_layers: Tuple[int, ...] = ALL_PREDICTION_LAYERS,
        backbone_weights: str = "",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.prediction_layers = tuple(prediction_layers)
        self.num_priors_per_layer = tuple(num_priors_per_layer)
        assert len(self.prediction_layers) == len(self.num_priors_per_layer), (
            "prediction_layers 与 num_priors_per_layer 长度必须一致"
        )
        self.num_priors = sum(
            FEATURE_MAP_SIZES[i] ** 2 * n
            for i, n in zip(self.prediction_layers, self.num_priors_per_layer)
        )

        self._build_backbone(pretrained, backbone_weights)

        # ---- conv4_3 的 L2 归一化（官方 normalizations[0] = 20）----
        self.l2_norm = L2Norm(512, l2_norm_scale)

        # ---- 预测头 ----
        # 每个被选中的预测层后接一个 3x3(pad=1) 卷积，分别输出 n*4 与 n*C 个通道。
        # 通道数由 prediction_layers 与 num_priors_per_layer 共同决定，
        # 必须与 build_prior_box() 产出的先验框顺序、数量严格一致。
        in_channels = [HEAD_IN_CHANNELS[i] for i in self.prediction_layers]
        self.loc_heads = nn.ModuleList(
            [
                nn.Conv2d(cin, n * 4, kernel_size=3, padding=1)
                for cin, n in zip(in_channels, self.num_priors_per_layer)
            ]
        )
        self.conf_heads = nn.ModuleList(
            [
                nn.Conv2d(cin, n * num_classes, kernel_size=3, padding=1)
                for cin, n in zip(in_channels, self.num_priors_per_layer)
            ]
        )
        _xavier_init(self.loc_heads)
        _xavier_init(self.conf_heads)

    # ----------------------------------------------------------------------------------
    # 主干构建
    # ----------------------------------------------------------------------------------
    def _build_backbone(self, pretrained: bool, backbone_weights: str = "") -> None:
        """构建 VGG16 主干。

        权重来源优先级：
            1. ``backbone_weights`` 指定的本地文件（离线可用，报告中记录确切文件）
            2. torchvision 的 ImageNet 权重自动下载（缓存于 ``$TORCH_HOME``）
            3. 随机初始化
        """
        import os

        if backbone_weights and os.path.isfile(backbone_weights):
            vgg = torchvision.models.vgg16(weights=None)
            state = torch.load(backbone_weights, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # 兼容「只含 features.*」与「纯 features 层键」两种格式
            prefixes = ("features.", "module.features.", "")
            cleaned = None
            for prefix in prefixes:
                if prefix and not any(k.startswith(prefix) for k in state):
                    continue
                candidate = {
                    (k[len(prefix):] if prefix else k): v
                    for k, v in state.items()
                    if (not prefix) or k.startswith(prefix)
                }
                if any(k.endswith("0.weight") for k in candidate):
                    cleaned = candidate
                    break
            if cleaned is None:
                cleaned = state
            missing, unexpected = vgg.features.load_state_dict(cleaned, strict=False)
            n_loaded = len(cleaned) - len(unexpected)
            print(
                f"主干权重 : 已从本地加载 {backbone_weights} "
                f"({n_loaded}/{len(vgg.features.state_dict())} 个张量)"
            )
            if unexpected:
                print(f"          未使用的键: {len(unexpected)} 个")
        elif pretrained:
            vgg = torchvision.models.vgg16(
                weights=torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES
            )
            print("主干权重 : 已加载 torchvision ImageNet 预训练 VGG16 features")
        else:
            vgg = torchvision.models.vgg16(weights=None)
            print("主干权重 : 随机初始化（未使用预训练）")

        backbone = vgg.features

        maxpool_positions = [i for i, layer in enumerate(backbone) if isinstance(layer, nn.MaxPool2d)]
        _, _, maxpool3_pos, maxpool4_pos, _ = maxpool_positions

        # 关键细节：把 pool3 改成 ceil_mode，才能得到论文的 38x38（300 -> 150 -> 75 -> 38）
        backbone[maxpool3_pos].ceil_mode = True

        # 到 relu4_3 为止（输出 38x38x512，即第一个预测层的输入）
        self.features = nn.Sequential(*backbone[:maxpool4_pos])

        # pool4 -> conv5_3 -> pool5(3x3 s1) -> conv6(空洞) -> conv7
        fc = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=False),  # pool5，步长 1
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),          # fc6 -> 空洞卷积
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),                                # fc7
            nn.ReLU(inplace=True),
        )
        _xavier_init(fc)

        self.extra = nn.ModuleList(
            [
                nn.Sequential(
                    *backbone[maxpool4_pos:-1],  # pool4 + conv5_1~relu5_3（跳过原 pool5）
                    fc,
                ),
                nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=1), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2), nn.ReLU(inplace=True),  # conv8_2 -> 10x10
                ),
                nn.Sequential(
                    nn.Conv2d(512, 128, kernel_size=1), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2), nn.ReLU(inplace=True),  # conv9_2 -> 5x5
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3), nn.ReLU(inplace=True),                        # conv10_2 -> 3x3
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3), nn.ReLU(inplace=True),                        # conv11_2 -> 1x1
                ),
            ]
        )
        # 注意：**只对新增层做 Xavier 初始化**。
        # extra[0] 里含有预训练的 conv5_1~conv5_3（来自 ImageNet 权重），
        # 若对整个 extra 做初始化会把它们覆盖成随机值，训练会立刻发散。
        for block in self.extra[1:]:
            _xavier_init(block)

    # ----------------------------------------------------------------------------------
    # 前向
    # ----------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 ``(loc, conf)``。

        loc:  (N, num_priors, 4)              相对先验框的偏移量
        conf: (N, num_priors, num_classes)    各类别得分（未过 softmax）

        先验框的顺序为：按 ``prediction_layers`` 顺序，每层内按「行 h 外层、
        列 w 内层、位置内 n 个先验框」展开 —— 与 :class:`PriorBox` 完全一致。
        """
        all_sources: List[torch.Tensor] = []

        # 源 0: conv4_3，先做 L2 归一化
        x = self.features(x)
        all_sources.append(self.l2_norm(x))

        # 源 1..5 对应 extra[0..4]。
        # 只跑到被选中的最靠后的那个预测层，多余的分支不再计算（消融时可省算力）。
        needed = max(self.prediction_layers)
        for block in self.extra[:needed]:
            x = block(x)
            all_sources.append(x)

        locs: List[torch.Tensor] = []
        confs: List[torch.Tensor] = []
        for k, (layer_idx, loc_head, conf_head) in enumerate(
            zip(self.prediction_layers, self.loc_heads, self.conf_heads)
        ):
            src = all_sources[layer_idx]
            loc = loc_head(src)    # (N, n*4, H, W)
            conf = conf_head(src)  # (N, n*C, H, W)
            n, _, h, w = loc.shape
            n_priors = self.num_priors_per_layer[k]

            # (N, n*4, H, W) -> (N, n, 4, H, W) -> (N, H, W, n, 4) -> (N, H*W*n, 4)
            loc = loc.view(n, n_priors, 4, h, w).permute(0, 3, 4, 1, 2).reshape(n, h * w * n_priors, 4)
            conf = (
                conf.view(n, n_priors, self.num_classes, h, w)
                .permute(0, 3, 4, 1, 2)
                .reshape(n, h * w * n_priors, self.num_classes)
            )

            locs.append(loc)
            confs.append(conf)

        return torch.cat(locs, dim=1), torch.cat(confs, dim=1)

    # ----------------------------------------------------------------------------------
    # 辅助：冻结/解冻主干（便于分阶段微调）
    # ----------------------------------------------------------------------------------
    def freeze_backbone(self, freeze: bool = True) -> None:
        for p in self.features.parameters():
            p.requires_grad = not freeze
        for p in self.extra[0].parameters():
            p.requires_grad = not freeze


def build_ssd300(
    num_classes: int = 21,
    pretrained: bool = True,
    backbone_weights: str = "",
    prediction_layers: Tuple[int, ...] = ALL_PREDICTION_LAYERS,
    num_priors_per_layer: Tuple[int, ...] = NUM_PRIORS_PER_LAYER,
) -> SSD300:
    """构造 SSD300。

    ``prediction_layers`` 与 ``num_priors_per_layer`` 必须与
    :func:`src.models.prior_box.build_prior_box` / :func:`model_head_spec`
    保持一致，推荐直接用 ``model_head_spec(cfg)`` 的结果传入。
    """
    return SSD300(
        num_classes=num_classes,
        pretrained=pretrained,
        backbone_weights=backbone_weights,
        prediction_layers=prediction_layers,
        num_priors_per_layer=num_priors_per_layer,
    )


if __name__ == "__main__":
    net = build_ssd300(pretrained=False)
    net.eval()
    with torch.no_grad():
        loc, conf = net(torch.zeros(1, 3, 300, 300))
    print(f"先验框总数      : {net.num_priors}")
    print(f"loc  输出形状   : {tuple(loc.shape)}")
    print(f"conf 输出形状   : {tuple(conf.shape)}")
    print(f"参数量          : {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M")
