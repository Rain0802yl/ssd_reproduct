#!/bin/bash
# ============================================================================
# SSD 复现 —— 环境变量设置（被其它脚本 source 使用）
#
# 作用：
#   1. 自动定位 Python 解释器
#        本地 macOS  -> 用项目自带的 env_ssd/bin/python
#        云服务器 Linux -> 用系统 python3（AUP 云已预装 ROCm 版 PyTorch）
#   2. 把所有缓存目录固定到本项目内，避免污染系统目录
#
# 用法：  source 脚本/环境变量.sh
#
# 注意：本脚本同时兼容 bash 与 zsh。
#       （macOS 的 VSCode 默认终端是 zsh，不能用 bash 专有的 BASH_SOURCE）
# ============================================================================

# ---------------------------------------------------------------------------
# 定位本脚本自身的路径
#   zsh  : $0 在 source 时就是脚本路径；更可靠的是 ${(%):-%x}
#   bash : 用 BASH_SOURCE[0]
# 用 eval 延迟解析，避免另一种 shell 在解析阶段报语法错
# ---------------------------------------------------------------------------
if [ -n "${ZSH_VERSION:-}" ]; then
    eval '_DSH_SRC="${(%):-%x}"'
else
    if [ -n "${BASH_SOURCE[0]:-}" ]; then
        _DSH_SRC="${BASH_SOURCE[0]}"
    else
        _DSH_SRC="$0"
    fi
fi

# 项目根目录 = 本脚本所在目录（脚本/）的上一级
PROJECT_ROOT="$(cd "$(dirname "$_DSH_SRC")/.." 2>/dev/null && pwd)"

# ---------------------------------------------------------------------------
# 安全校验：定位到的目录必须真的是本项目
# ---------------------------------------------------------------------------
if [ -z "$PROJECT_ROOT" ] || [ ! -f "$PROJECT_ROOT/src/train.py" ]; then
    echo "[环境] ❌ 错误：无法定位项目根目录" >&2
    echo "[环境]    解析结果：${PROJECT_ROOT:-<空>}" >&2
    echo "[环境]    请先 cd 到项目根目录，再执行： source 脚本/环境变量.sh" >&2
    return 1 2>/dev/null || exit 1
fi

cd "$PROJECT_ROOT" || { return 1 2>/dev/null || exit 1; }

# ---------------------------------------------------------------------------
# Python 解释器：优先项目自带环境（macOS），否则回退系统 python3（Linux 云）
# ---------------------------------------------------------------------------
if [ -x "$PROJECT_ROOT/env_ssd/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/env_ssd/bin/python"
    PYTHON_SRC="项目自带环境 env_ssd（macOS）"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
    PYTHON_SRC="系统 python3（Linux / 云服务器）"
elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
    PYTHON_SRC="系统 python"
else
    echo "[环境] ❌ 错误：找不到可用的 Python 解释器（python3 / python 都不存在）" >&2
    return 1 2>/dev/null || exit 1
fi

# ---------------------------------------------------------------------------
# 缓存目录固定到项目内
#   注意：Linux 云上 .cache/torch 可能不存在（未上传），
#         此时训练脚本会自动回退到 torchvision 在线下载 VGG16 权重。
# ---------------------------------------------------------------------------
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
mkdir -p "$TORCH_HOME" "$MPLCONFIGDIR" 2>/dev/null

echo "[环境] 项目根目录 : $PROJECT_ROOT"
echo "[环境] Python     : $PYTHON"
echo "[环境] 来源       : $PYTHON_SRC"
echo "[环境] TORCH_HOME : $TORCH_HOME"
