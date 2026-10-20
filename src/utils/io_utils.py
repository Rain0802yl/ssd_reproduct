"""跨平台文件读写工具。

**为什么需要这个模块**

OpenCV 的 ``cv2.imread`` / ``cv2.imwrite`` 在 Linux（以及部分 Windows 构建）上
**不支持非 ASCII 路径** —— 遇到中文目录名会静默返回 ``None`` 或写入失败。

本项目的数据集目录名是中文（``数据集/VOCdevkit/VOC2007/...``），
在 macOS 上可以正常工作，但上传到 Linux 云服务器后 ``cv2.imread`` 会读不到图。

解决办法：先用 Python 自带的文件 IO 读成字节流，再交给 OpenCV 解码 / 编码。
这样路径处理完全由 Python 负责，与平台和 locale 无关。
"""

from __future__ import annotations

import os
from typing import Any, Optional

import cv2
import numpy as np


def imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """读取图像，支持中文等非 ASCII 路径。

    行为与 ``cv2.imread`` 一致：读取失败返回 ``None``。
    """
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, flags)
    return img


def imwrite_unicode(path: str, img: np.ndarray, params: Optional[Any] = None) -> bool:
    """保存图像，支持中文等非 ASCII 路径。

    行为与 ``cv2.imwrite`` 一致：成功返回 ``True``，失败返回 ``False``。
    """
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img, params if params is not None else [])
    if not ok:
        return False
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        buf.tofile(path)
    except OSError:
        return False
    return True
