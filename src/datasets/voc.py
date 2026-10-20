"""PASCAL VOC 检测数据集。

**对齐官方 Caffe 实现** ``src/caffe/util/io.cpp : ReadXMLToAnnotatedDatum``（第 258-358 行）：

    * 标注框从 XML 的 ``<bndbox>`` 读取**绝对像素坐标** ``xmin/ymin/xmax/ymax``
    * 然后**除以图像宽高**得到归一化坐标（第 348-351 行）
    * ``<difficult>`` 标志原样保留（训练时 ``use_difficult_gt=True`` 会使用它们，
      评测时 ``evaluate_difficult_gt=False`` 会忽略它们）
    * 官方代码**不做 -1 偏移**，也不把框裁剪到图像边界（只打警告），此处保持一致

类别顺序取自 ``data/VOC0712/labelmap_voc.prototxt``（0 为背景），与论文 Table 1 的
20 类顺序一致。
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# 与官方 labelmap_voc.prototxt 完全一致
VOC_CLASSES: Tuple[str, ...] = (
    "__background__",  # 0
    "aeroplane", "bicycle", "bird", "boat", "bottle",      # 1-5
    "bus", "car", "cat", "chair", "cow",                   # 6-10
    "diningtable", "dog", "horse", "motorbike", "person",  # 11-15
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",  # 16-20
)
NUM_CLASSES = len(VOC_CLASSES)  # 21


class VOCDetection(Dataset):
    """VOC2007 / VOC2012 检测数据集。

    参数:
        voc_root:   包含 ``VOC2007`` 目录的根目录（即 VOCdevkit）
        year:       '2007' 或 '2012'
        image_set:  'trainval' / 'test' / 'train' / 'val'
        transform:  变换函数，签名 ``(img_bgr_hwc_uint8, boxes, labels) -> (img, boxes, labels)``
    """

    def __init__(
        self,
        voc_root: str,
        year: str = "2007",
        image_set: str = "trainval",
        transform: Optional[Callable] = None,
        class_names: Tuple[str, ...] = VOC_CLASSES,
    ) -> None:
        self.voc_root = voc_root
        self.year = year
        self.image_set = image_set
        self.transform = transform
        self.class_to_ind = {name: i for i, name in enumerate(class_names)}

        self.root = os.path.join(voc_root, f"VOC{year}")
        ids_file = os.path.join(self.root, "ImageSets", "Main", f"{image_set}.txt")
        if not os.path.isfile(ids_file):
            raise FileNotFoundError(f"找不到数据集划分文件: {ids_file}")
        with open(ids_file, "r") as f:
            self.ids = [line.strip() for line in f if line.strip()]

        self._degenerate_boxes = 0  # 统计被丢弃的退化框，便于报告中如实说明

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def degenerate_boxes(self) -> int:
        """被跳过的退化框（xmax<=xmin 或 ymax<=ymin）数量。"""
        return self._degenerate_boxes

    # ----------------------------------------------------------------------------------
    def _load_annotation(self, index: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """解析 XML，返回归一化框、类别下标、difficult 标志。"""
        anno_path = os.path.join(self.root, "Annotations", f"{index}.xml")
        tree = ET.parse(anno_path)
        root = tree.getroot()
        size = root.find("size")
        width = float(size.find("width").text)
        height = float(size.find("height").text)

        boxes: List[List[float]] = []
        labels: List[int] = []
        difficults: List[int] = []

        for obj in root.findall("object"):
            name_node = obj.find("name")
            if name_node is None:
                continue
            name = name_node.text.strip()
            if name not in self.class_to_ind:
                continue

            diff_node = obj.find("difficult")
            difficult = int(diff_node.text) if diff_node is not None else 0

            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            # 跳过退化框：官方代码在 EncodeBBox 中会 CHECK_GT(width, 0) 直接崩溃，
            # 这里改为跳过，并在数据集中计数以便报告如实说明。
            if xmax <= xmin or ymax <= ymin:
                self._degenerate_boxes += 1
                continue

            # 与官方一致：不做 -1 偏移，不裁剪到图像边界
            boxes.append([xmin / width, ymin / height, xmax / width, ymax / height])
            labels.append(self.class_to_ind[name])
            difficults.append(difficult)

        if not boxes:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
            )
        return (
            np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.int64),
            np.array(difficults, dtype=np.int64),
        )

    # ----------------------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict:
        img_id = self.ids[idx]
        img_path = os.path.join(self.root, "JPEGImages", f"{img_id}.jpg")
        # 用 OpenCV 读取，得到 BGR 顺序，与官方 DecodeDatumToCVMatNative 一致
        # 用 imread_unicode 而非 cv2.imread：Linux 上 cv2.imread 不支持中文路径
        from ..utils.io_utils import imread_unicode

        img = imread_unicode(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"无法读取图像: {img_path}")

        boxes, labels, difficults = self._load_annotation(img_id)
        orig_h, orig_w = img.shape[:2]

        if self.transform is not None:
            img, boxes, labels = self.transform(img, boxes, labels)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(img)),
            "boxes": torch.from_numpy(np.asarray(boxes, dtype=np.float32)),
            "labels": torch.from_numpy(np.asarray(labels, dtype=np.int64)),
            "difficult": torch.from_numpy(difficults),
            "id": img_id,
            "orig_size": (orig_h, orig_w),
        }


def detection_collate(batch: List[Dict]):
    """批处理：图像堆叠，框/标签保持为变长列表。"""
    images = torch.stack([b["image"] for b in batch], dim=0)
    boxes = [b["boxes"] for b in batch]
    labels = [b["labels"] for b in batch]
    ids = [b["id"] for b in batch]
    orig_sizes = [b["orig_size"] for b in batch]
    difficults = [b["difficult"] for b in batch]
    return images, boxes, labels, ids, orig_sizes, difficults
