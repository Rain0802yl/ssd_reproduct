"""自测脚本：在正式训练前验证各模块的正确性与性能。

运行::

    ./env_ssd/bin/python -m src.selftest

覆盖内容:
    1. 先验框数量与论文一致（8732）
    2. 框编码/解码可逆
    3. IoU 计算正确
    4. 匹配策略正确（正样本 / 难负样本挖掘）
    5. 数据集与数据增强正常
    6. 单图过拟合（loss 能否快速下降 —— 检验整条链路可训练）
    7. MPS 上的实际训练速度
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .config import load_config
from .datasets.augment import SSDTransform
from .datasets.voc import VOCDetection
from .models.loss import MultiBoxLoss, match_priors_to_gt
from .models.prior_box import build_prior_box, build_ssd300_prior_box, model_head_spec
from .models.vgg_ssd import build_ssd300
from .train import pick_device
from .utils.box_utils import decode_bbox, encode_bbox, jaccard_overlap

PASS, FAIL = "  [通过]", "  [失败]"
results = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((name, cond))
    print(f"{PASS if cond else FAIL} {name}" + (f"  {detail}" if detail else ""))


def main() -> int:
    print("=" * 78)
    print("SSD300 复现 —— 模块自测")
    print("=" * 78)

    # ---------------- 1. 先验框 ----------------
    print("\n[1] 先验框生成")
    pb = build_ssd300_prior_box()
    priors = pb()
    check("先验框总数为 8732", priors.shape[0] == 8732, f"实际 {priors.shape[0]}")
    check(
        "每层每位置的框数正确",
        pb.num_priors_per_layer == [4, 6, 6, 6, 4, 4],
        f"实际 {pb.num_priors_per_layer}",
    )
    check("先验框形状为 (8732, 4)", tuple(priors.shape) == (8732, 4))
    check("先验框中心合理性", bool((priors[:, 2] > priors[:, 0]).all()))

    # ---------------- 2. 编码/解码可逆 ----------------
    print("\n[2] 框编码 / 解码")
    var = torch.tensor([0.1, 0.1, 0.2, 0.2])
    gt = torch.tensor([[0.2, 0.3, 0.6, 0.8], [0.1, 0.1, 0.4, 0.5]])
    pr = priors[:2]
    enc = encode_bbox(pr, gt, var)
    dec = decode_bbox(pr, enc, var)
    check("编码->解码可还原真值框", bool(torch.allclose(dec, gt, atol=1e-5)),
          f"最大误差 {float((dec - gt).abs().max()):.2e}")

    # ---------------- 3. IoU ----------------
    print("\n[3] IoU 计算")
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    b = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 1.5, 1.5], [2.0, 2.0, 3.0, 3.0]])
    iou = jaccard_overlap(a, b)[0]
    check("完全重合 IoU = 1", abs(float(iou[0]) - 1.0) < 1e-6, f"{float(iou[0]):.4f}")
    check("四分之一重合 IoU = 0.25/1.75", abs(float(iou[1]) - 0.25 / 1.75) < 1e-6,
          f"{float(iou[1]):.4f}")
    check("不相交 IoU = 0", abs(float(iou[2])) < 1e-6, f"{float(iou[2]):.4f}")

    # ---------------- 4. 匹配与难负样本挖掘 ----------------
    print("\n[4] 匹配策略与难负样本挖掘")
    p = torch.tensor(
        [
            [0.0, 0.0, 0.2, 0.2],   # 与 GT1 高度重合
            [0.8, 0.8, 1.0, 1.0],   # 与 GT2 高度重合
            [0.5, 0.0, 0.55, 0.05],  # 与任何 GT 几乎无重叠
        ]
    )
    g = torch.tensor([[0.0, 0.0, 0.2, 0.2], [0.8, 0.8, 1.0, 1.0]])
    idx, ov = match_priors_to_gt(p, g, 0.5)
    check("先验框 0 匹配到 GT 0", int(idx[0]) == 0, f"match_idx={idx.tolist()}")
    check("先验框 1 匹配到 GT 1", int(idx[1]) == 1)
    check("先验框 2 未匹配（负样本）", int(idx[2]) == -1)
    check("负样本重叠度 < 0.5", float(ov[2]) < 0.5, f"{float(ov[2]):.4f}")

    # 一个 GT 强制匹配（即使 IoU 很低，但只要有重叠就会被强制匹配）
    p2 = torch.tensor([[0.0, 0.0, 0.1, 0.1]])
    g2 = torch.tensor([[0.05, 0.05, 1.0, 1.0]])
    idx2, _ = match_priors_to_gt(p2, g2, 0.5)
    check("每个 GT 至少被一个先验框匹配（强制匹配）", int(idx2[0]) == 0,
          "注：完全零重叠时官方也不匹配（Caffe 只保留 overlap>1e-6 的候选对）")

    # ---------------- 5. 数据集与增强 ----------------
    print("\n[5] 数据集与数据增强")
    cfg = load_config("src/configs/ssd300_voc_quick.yaml")
    a = cfg["augment"]
    tf = SSDTransform(resize=300, mean_bgr=a["mean_bgr"], train=True, seed=42, norm_mode=a.get("norm_mode","imagenet"))
    ds = VOCDetection(cfg["data"]["voc_root"], "2007", "trainval", transform=tf)
    check("训练集规模为 5011", len(ds) == 5011, f"实际 {len(ds)}")

    sample = ds[0]
    check("图像张量形状为 (3,300,300)", tuple(sample["image"].shape) == (3, 300, 300),
          f"实际 {tuple(sample['image'].shape)}")
    check("图像数据类型为 float32", sample["image"].dtype == torch.float32)
    check("框与标签数量一致", sample["boxes"].shape[0] == sample["labels"].shape[0])

    # 增强的随机性
    ds2 = VOCDetection(cfg["data"]["voc_root"], "2007", "trainval",
                       transform=SSDTransform(resize=300, train=True, seed=1))
    n_boxes = [ds2[i]["boxes"].shape[0] for i in range(20)]
    check("数据增强产生了不同的框数量（随机性正常）", len(set(n_boxes)) > 1,
          f"前 20 张框数={n_boxes}")

    # 测试变换（无增强）应保持框不变
    tf_test = SSDTransform(resize=300, mean_bgr=a["mean_bgr"], train=False)
    ds_test = VOCDetection(cfg["data"]["voc_root"], "2007", "test", transform=tf_test)
    check("测试集规模为 4952", len(ds_test) == 4952, f"实际 {len(ds_test)}")

    # ---------------- 6. 单图过拟合（检验整条链路可训练）----------------
    print("\n[6] 单图过拟合测试（约 30 秒）")
    device = pick_device("auto")
    print(f"      计算设备: {device}")
    torch.manual_seed(0)

    model = build_ssd300(num_classes=21, pretrained=False).to(device)
    priors_d = priors.to(device)
    variances_d = pb.variances.to(device)
    criterion = MultiBoxLoss(priors=priors_d, variances=variances_d, num_classes=21).to(device)
    # 注意：这里刻意用比官方更小的学习率（1e-4 而非 1e-3）。
    # 官方 1e-3 是针对 batch=32 的；单图过拟合时梯度方差极大，会梯度爆炸。
    # 本项检查的目的是验证「整条前向/损失/反向链路可训练」，而非验证超参数。
    opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)

    tf_of = SSDTransform(resize=300, mean_bgr=a["mean_bgr"], train=True, seed=0)
    ds_of = VOCDetection(cfg["data"]["voc_root"], "2007", "trainval", transform=tf_of)
    s = ds_of[0]
    img = s["image"].unsqueeze(0).to(device)
    bx = [s["boxes"].to(device)]
    lb = [s["labels"].to(device)]
    print(f"      该图有 {s['boxes'].shape[0]} 个目标, 类别 {s['labels'].tolist()}")

    losses = []
    model.train()
    t0 = time.time()
    for step in range(40):
        loc_p, conf_p = model(img)
        loss, st = criterion(loc_p, conf_p, bx, lb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(st["loss"])
    dt = time.time() - t0
    check(
        "单图过拟合：损失显著下降",
        losses[-1] < losses[0] * 0.5,
        f"{losses[0]:.4f} -> {losses[-1]:.4f}（{dt/40*1000:.0f} ms/步）",
    )
    check("正样本被正确匹配", st["num_pos"] > 0, f"num_pos={st['num_pos']}")

    # ---------------- 7. 训练速度 ----------------
    print("\n[7] 训练速度实测（batch=16）")
    model.train()
    x = torch.randn(16, 3, 300, 300, device=device)
    gts = [torch.tensor([[0.2, 0.3, 0.6, 0.8]], device=device) for _ in range(16)]
    lbs = [torch.tensor([1], device=device) for _ in range(16)]
    opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    for _ in range(2):  # 热身
        loc_p, conf_p = model(x)
        loss, _ = criterion(loc_p, conf_p, gts, lbs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device.type == "mps":
        torch.mps.synchronize()

    ts = []
    for _ in range(5):
        t = time.time()
        loc_p, conf_p = model(x)
        loss, _ = criterion(loc_p, conf_p, gts, lbs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if device.type == "mps":
            torch.mps.synchronize()
        ts.append(time.time() - t)
    avg = sum(ts) / len(ts)
    print(f"      完整训练步（含损失与匹配）: {avg:.2f} 秒/迭代")
    print(f"      -> 8000 迭代预计: {avg*8000/3600:.2f} 小时")
    print(f"      -> 20000 迭代预计: {avg*20000/3600:.2f} 小时")
    check("单步训练时间可接受（< 3 秒）", avg < 3.0, f"{avg:.2f} 秒/迭代")

    # ---------------- 8. mAP 评估器正确性 ----------------
    print("\n[8] VOC mAP 评估器（构造用例验证）")
    from .utils.voc_eval import evaluate_detections

    rng = np.random.default_rng(0)
    n_img = 40
    gts, perfect, half, empty = [], [], [], []
    for _ in range(n_img):
        k = int(rng.integers(1, 4))
        b = rng.random((k, 4)) * 0.6 + 0.2
        b = np.stack(
            [np.minimum(b[:, 0], b[:, 2]), np.minimum(b[:, 1], b[:, 3]),
             np.maximum(b[:, 0], b[:, 2]), np.maximum(b[:, 1], b[:, 3])], axis=1,
        )
        lb = rng.integers(1, 21, k)
        gts.append((b, lb, np.zeros(k, dtype=np.int64)))
        perfect.append((lb.copy(), np.full(k, 0.9), b.copy()))
        empty.append((np.zeros(0, dtype=int), np.zeros(0), np.zeros((0, 4))))
        if len(gts) % 2 == 0:
            half.append((lb.copy(), np.full(k, 0.9), b.copy()))
        else:
            rb = np.clip(rng.random((k, 4)) * 0.3, 0, 1)
            rb = np.stack([rb[:, 0], rb[:, 1], rb[:, 0] + 0.05, rb[:, 1] + 0.05], axis=1)
            half.append((lb.copy(), np.full(k, 0.3), rb))

    m_perfect = evaluate_detections(perfect, gts, num_classes=21)
    m_half = evaluate_detections(half, gts, num_classes=21)
    m_empty = evaluate_detections(empty, gts, num_classes=21)

    check("完美预测的 mAP 应为 100%", abs(m_perfect["mAP"] - 1.0) < 1e-6,
          f"实际 {m_perfect['mAP']*100:.2f}%")
    check("一半正确时 mAP 介于 30%~70%", 0.30 < m_half["mAP"] < 0.70,
          f"实际 {m_half['mAP']*100:.2f}%")
    check("空检测的 mAP 应为 0%", m_empty["mAP"] == 0.0, f"实际 {m_empty['mAP']*100:.2f}%")

    # ---------------- 9. 消融实验所需的架构开关 ----------------
    print("\n[9] 消融实验配置（先验框与网络头必须严格一致）")
    base_cfg = load_config("src/configs/ssd300_voc_quick.yaml")

    variants = [
        ("官方 6 层", {}),
        ("A2 去掉多形状先验框", {"model": {"aspect_ratios": [[], [], [], [], [], []]}}),
        ("A3 仅 conv4_3+conv7", {"model": {"prediction_layers": [0, 1]}}),
        ("A3b 仅 conv4_3", {"model": {"prediction_layers": [0]}}),
    ]
    all_ok = True
    for name, patch in variants:
        cfg_v = load_config("src/configs/ssd300_voc_quick.yaml")
        for sect, kv in patch.items():
            cfg_v[sect].update(kv)
        pb_v = build_prior_box(cfg_v)
        spec_v = model_head_spec(cfg_v)
        n_priors = pb_v().shape[0]
        net_v = build_ssd300(num_classes=21, pretrained=False, **spec_v).eval()
        with torch.no_grad():
            loc_v, conf_v = net_v(torch.zeros(1, 3, 300, 300))
        ok = (
            n_priors == loc_v.shape[1]
            and n_priors == net_v.num_priors
            and conf_v.shape[1] == n_priors
        )
        all_ok &= ok
        print(f"      {name:22s} 先验框={n_priors:5d}  网络输出={loc_v.shape[1]:5d}  {'OK' if ok else '不一致!'}")
    check("四种消融设置的先验框数都与网络输出一致", all_ok)

    # 关闭难负样本挖掘时，损失必须仍然是有限值
    cfg_m = load_config("src/configs/ablation_A4_关闭难负样本挖掘.yaml")
    crit_m = MultiBoxLoss(
        priors=priors,
        variances=pb.variances,
        num_classes=21,
        mining_type=cfg_m["loss"]["mining_type"],
        normalize_conf_by_samples=cfg_m["loss"]["normalize_conf_by_samples"],
    )
    torch.manual_seed(0)
    net_m = build_ssd300(num_classes=21, pretrained=False).eval()
    with torch.no_grad():
        loc_m, conf_m = net_m(torch.zeros(2, 3, 300, 300))
    g_m = [torch.tensor([[0.2, 0.3, 0.6, 0.8]]), torch.tensor([[0.1, 0.1, 0.5, 0.9]])]
    l_m = [torch.tensor([1]), torch.tensor([7])]
    loss_m, st_m = crit_m(loc_m, conf_m, g_m, l_m)
    check(
        "关闭难负样本挖掘时损失仍为有限值",
        bool(torch.isfinite(loss_m)),
        f"loss={float(loss_m):.4f}, 分类样本数={st_m['num_conf_samples']}（远多于正样本 {st_m['num_pos']}）",
    )

    # ---------------- 汇总 ----------------
    n_pass = sum(1 for _, c in results if c)
    print("\n" + "=" * 78)
    print(f"自测结果: {n_pass}/{len(results)} 项通过")
    if n_pass < len(results):
        print("未通过项:")
        for name, c in results:
            if not c:
                print(f"    - {name}")
    print("=" * 78)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
