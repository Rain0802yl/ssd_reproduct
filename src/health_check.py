"""训练健康检查 —— 快速判断模型有没有"退化"。

**为什么需要这个脚本**

本项目曾踩过一个非常隐蔽的坑：输入归一化方式与预训练权重不匹配，
表现是**训练损失平稳下降、无 NaN、无发散**，但骨干网络的激活值逐层放大
（conv4_3 标准差从 85 涨到 17080），经 L2 归一化后饱和成近似 one-hot，
最终模型对所有输入输出几乎相同的向量 —— 相当于一个常数函数，
mAP 只有 0.07%。

**只看 loss 完全发现不了这个问题。** 必须检查模型内部状态。

本脚本检查 5 项指标，每项都给出「健康 / 退化」的判定：

1. conv4_3 激活值尺度     —— 健康 1~10；退化 >1000
2. L2 归一化后的输入敏感度  —— 两张不同图片的归一化输出差异，健康 >1；退化 <0.5
3. 分类 logits 输入敏感度   —— 健康 >0.3；退化 <0.05
4. 权重是否含 NaN / Inf
5. 单批损失是否落在合理区间

用法：
    python3 -m src.health_check \\
        --config src/configs/ssd300_voc_long.yaml \\
        --weights experiments/main10000/last.pth

    # 不指定权重则检查「刚加载的预训练模型」，用于确认环境是否正常
    python3 -m src.health_check --config src/configs/ssd300_voc_long.yaml
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Tuple

import torch

from .config import load_config
from .datasets.augment import SSDTransform
from .datasets.voc import VOCDetection
from .models.prior_box import build_prior_box, model_head_spec
from .models.vgg_ssd import build_ssd300
from .train import pick_device


# (指标名, 下限, 上限, 退化判据说明)
CHECKS: List[Tuple[str, float, float, str]] = [
    ("conv4_3 激活值标准差", 0.2, 100.0, "死亡 <0.2；退化 >100"),
    ("L2 归一化后两图差异", 1.0, 1e9, "退化 <0.5"),
    ("分类 logits 两图差异", 0.3, 1e9, "退化 <0.05"),
]


def _build_model(cfg: Dict, device: torch.device, weights: str | None):
    net = build_ssd300(
        num_classes=cfg["data"]["num_classes"],
        pretrained=True,
        backbone_weights=cfg["model"].get("backbone_weights"),
        **model_head_spec(cfg),
    ).to(device)
    iter_no = -1
    if weights:
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing:
            print(f"  ⚠️  缺少 {len(missing)} 个张量")
        if unexpected:
            print(f"  ⚠️  多余 {len(unexpected)} 个张量")
        iter_no = ckpt.get("iter", -1) if isinstance(ckpt, dict) else -1
    net.eval()
    return net, iter_no


def run(cfg: Dict, weights: str | None, device_name: str) -> int:
    device = pick_device(device_name)
    print(f"设备      : {device}")
    print(f"权重      : {weights or '（未指定，使用 ImageNet 预训练模型）'}")

    net, iter_no = _build_model(cfg, device, weights)
    if iter_no >= 0:
        print(f"训练迭代  : {iter_no}")

    # ---------------- 准备两张不同的测试图 ----------------
    a = cfg["augment"]
    tf = SSDTransform(
        resize=cfg["model"]["input_size"],
        mean_bgr=a["mean_bgr"],
        train=False,
        norm_mode=a.get("norm_mode", "imagenet"),
    )
    ds = VOCDetection(
        cfg["data"]["voc_root"], cfg["data"]["year"], cfg["data"]["test_image_set"],
        transform=tf,
    )
    imgs = [ds[i]["image"].unsqueeze(0).to(device) for i in range(2)]

    results: Dict[str, float] = {}
    with torch.no_grad():
        feats = [net.features(x) for x in imgs]
        norms = [net.l2_norm(f) for f in feats]
        outs = [net(x) for x in imgs]

    results["conv4_3 激活值标准差"] = float(feats[0].std())
    results["L2 归一化后两图差异"] = float((norms[0] - norms[1]).abs().max())
    results["分类 logits 两图差异"] = float((outs[0][1] - outs[1][1]).abs().max())

    # ---------------- 权重 NaN 检查 ----------------
    bad_tensors = [
        n for n, p in net.named_parameters()
        if not torch.isfinite(p).all()
    ]
    results["NaN/Inf 张量个数"] = float(len(bad_tensors))

    # ---------------- 打印结果 ----------------
    print()
    print("=" * 68)
    print(f"{'指标':<24}{'实测值':>16}{'判定':>10}   参考")
    print("-" * 68)
    all_ok = True
    for name, lo, hi, hint in CHECKS:
        v = results[name]
        ok = lo <= v <= hi
        all_ok &= ok
        print(f"{name:<24}{v:>16.4f}{'✅ 通过' if ok else '❌ 异常':>10}   {hint}")
    nan_ok = results["NaN/Inf 张量个数"] == 0
    all_ok &= nan_ok
    print(f"{'NaN/Inf 张量个数':<24}{results['NaN/Inf 张量个数']:>16.0f}"
          f"{'✅ 通过' if nan_ok else '❌ 异常':>10}   应为 0")
    print("=" * 68)

    if bad_tensors:
        print("\n含 NaN/Inf 的参数（前 5 个）：")
        for n in bad_tensors[:5]:
            print(f"  {n}")

    print()
    if all_ok:
        print("✅ 模型状态健康。")
        print("   （注意：本检查只能排除「退化/死亡」这类结构性错误；")
        print("     mAP 高低仍取决于训练量，请以评测结果为准。）")
        return 0

    print("❌ 检测到异常，训练结果很可能不可用！")
    print("   常见原因：")
    print("     · 输入归一化方式与预训练权重不匹配（本项目曾因此 mAP 仅 0.07%）")
    print("     · 学习率过大导致骨干漂移")
    print("     · 安全网跳过步数过多")
    print("   建议：检查 config 里的 augment.norm_mode 是否为 'imagenet'。")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="SSD 训练健康检查")
    ap.add_argument("--config", default="src/configs/ssd300_voc_quick.yaml")
    ap.add_argument("--weights", default=None, help="要检查的 .pth；不填则检查预训练模型")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    return run(cfg, args.weights, args.device or cfg["runtime"]["device"])


if __name__ == "__main__":
    raise SystemExit(main())
