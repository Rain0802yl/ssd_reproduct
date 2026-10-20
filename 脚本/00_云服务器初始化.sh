#!/bin/bash
# ============================================================================
# 云服务器 / 新机器 一键初始化
#
# 做四件事：
#   ① 检查 Python 与 PyTorch（含 ROCm / CUDA / MPS 后端）
#   ② 检查依赖包，缺什么提示装什么
#   ③ 检查 VOC2007 数据集，缺了自动下载解压（约 911MB）
#   ④ 打印接下来的训练命令
#
# 用法：  bash 脚本/00_云服务器初始化.sh
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/环境变量.sh" || exit 1

echo
echo "==================== ① 运行时检查 ===================="
"$PYTHON" - <<'PY'
import sys, platform
print(f"  Python      : {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")
try:
    import torch
except ImportError:
    print("  ❌ PyTorch 未安装！")
    print("     云服务器一般已预装 ROCm 版，若确实没有请找管理员或按官方文档安装。")
    raise SystemExit(1)
print(f"  PyTorch     : {torch.__version__}")
hip = getattr(torch.version, "hip", None)
print(f"  HIP (ROCm)  : {hip if hip else '无'}")
print(f"  CUDA 可用   : {torch.cuda.is_available()}")
print(f"  MPS 可用    : {torch.backends.mps.is_available() if hasattr(torch.backends,'mps') else False}")
if torch.cuda.is_available():
    print(f"  GPU 设备    : {torch.cuda.get_device_name(0)}")
    print(f"  显存        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    backend = "ROCm (AMD)" if hip else "CUDA (NVIDIA)"
    print(f"  加速后端    : {backend}")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("  加速后端    : MPS (Apple GPU)")
else:
    print("  ⚠️  只有 CPU！训练会非常慢，请确认显卡驱动/PyTorch 版本。")
PY

echo
echo "==================== ② 依赖检查 ===================="
"$PYTHON" - <<'PY'
import importlib
need = ["torch", "torchvision", "cv2", "matplotlib", "numpy", "yaml", "tqdm", "PIL"]
missing = []
for m in need:
    try:
        importlib.import_module(m)
        print(f"  ✅ {m}")
    except Exception:
        print(f"  ❌ {m}  缺失")
        missing.append({"cv2": "opencv-python-headless", "yaml": "pyyaml", "PIL": "pillow"}.get(m, m))
if missing:
    print()
    print("  缺少依赖，请执行（不要重装 torch/torchvision！）：")
    print(f"    python3 -m pip install --user {' '.join(missing)}")
else:
    print("\n  ✅ 依赖齐全")
PY

echo
echo "==================== ③ 数据集检查 ===================="
VOC="数据集/VOCdevkit/VOC2007"
if [ -f "$VOC/ImageSets/Main/trainval.txt" ] && [ -f "$VOC/ImageSets/Main/test.txt" ]; then
    echo "  ✅ 数据集已就绪"
    echo "     trainval: $(wc -l < "$VOC/ImageSets/Main/trainval.txt" | tr -d ' ') 张"
    echo "     test    : $(wc -l < "$VOC/ImageSets/Main/test.txt" | tr -d ' ') 张"
else
    echo "  ⚠️  数据集缺失，开始下载（约 911MB，需要联网）..."
    mkdir -p 数据集 && cd 数据集 || exit 1
    for f in VOCtrainval_06-Nov-2007.tar VOCtest_06-Nov-2007.tar; do
        if [ ! -f "$f" ]; then
            echo "     下载 $f ..."
            curl -L --retry 3 -o "$f" "https://pjreddie.com/media/files/$f" || { echo "  ❌ 下载失败: $f"; exit 1; }
        fi
    done
    echo "     解压 ..."
    tar -xf VOCtrainval_06-Nov-2007.tar
    tar -xf VOCtest_06-Nov-2007.tar
    cd "$PROJECT_ROOT" || exit 1
    if [ -f "$VOC/ImageSets/Main/trainval.txt" ]; then
        echo "  ✅ 数据集下载解压完成"
        ls -1 "$VOC"
    else
        echo "  ❌ 解压后仍找不到 $VOC，请手动检查"; exit 1
    fi
fi

echo
echo "==================== ④ 下一步 ===================="
cat <<'TIP'
  快速验证训练（约 5 分钟，确认能跑通）：
      python3 -m src.train --config src/configs/ssd300_voc_quick.yaml \
          --max-iter 100 --exp-name smoke

  正式长训练（15000 迭代）：
      python3 -m src.train --config src/configs/ssd300_voc_long.yaml

  后台跑（断开连接也不中断）：
      nohup python3 -m src.train --config src/configs/ssd300_voc_long.yaml \
          > train.log 2>&1 &
      tail -f train.log

  评测：
      python3 -m src.eval --config src/configs/ssd300_voc_long.yaml \
          --weights experiments/long15000/last.pth

TIP
