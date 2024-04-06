from typing import Tuple

import numpy as np
import cv2
import torch
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


class CDNet2014Preprocess:
    # label definition: https://openaccess.thecvf.com/content_cvpr_workshops_2014/W12/papers/Wang_CDnet_2014_An_2014_CVPR_paper.pdf
    # * Define gt pixel value meaning
    PXL_VAL_STATIC = 0
    PXL_VAL_SHADOW = 50
    PXL_VAL_NONROI = 85
    PXL_VAL_UNKNOWN = 170
    PXL_VAL_MOVING = 255

    def __init__(self, image_size: Tuple[int] | None = None, isShadowFG=False, eps=5) -> None:
        self.image_size = image_size
        self.isShadowFG = isShadowFG
        self.eps = eps

    def __call__(self, gt: torch.Tensor) -> torch.Tensor:
        label = gt if self.image_size == None else TF.resize(gt, self.image_size, InterpolationMode.NEAREST)
        label = label.type(torch.float32)

        label[label < self.PXL_VAL_SHADOW + self.eps] = 0.0
        if self.isShadowFG:
            label[torch.where((label <= self.PXL_VAL_SHADOW + self.eps) & (label >= self.PXL_VAL_SHADOW - self.eps))] = 1.0

        label[label > self.PXL_VAL_MOVING - self.eps] = 1.0
        label[label > 1.0] = -1.0

        return label


class LASIESTAPreprocess:
    # label definition: https://www.gti.ssr.upm.es/data/lasiesta_database
    # * Define gt pixel value meaning
    # (R,G,B)
    MOVING2STATIC = (255, 255, 255)
    STATIC = (0, 0, 0)
    UNKNOWN = (128, 128, 128)  # Belong to NON roi

    OBJ1 = (255, 0, 0)  # Grayscale : 76
    OBJ2 = (0, 255, 0)  # Grayscale : 150
    OBJ3 = (255, 255, 0)  # Grayscale : 226

    def __init__(self, image_size: Tuple[int] | None = None, eps=5) -> None:
        self.eps = eps
        self.image_size = image_size

    def __call__(self, gt: np.ndarray) -> np.ndarray:
        '''gt: 2D image (Grayscale)'''
        label = gt.copy().astype(np.float32)
        cvtGray = np.array([0.299, 0.587, 0.114])
        OBJs_gray = np.array([self.OBJ1, self.OBJ2, self.OBJ3]) @ cvtGray

        fg_area = np.zeros_like(label).astype(bool)
        for gv in OBJs_gray:
            print(f'Low : {gv - self.eps} | High : {gv + self.eps}')
            fg_area = fg_area | ((label >= gv - self.eps) & (label <= gv + self.eps))

        label[label >= self.MOVING2STATIC[0] - self.eps] = 0.0
        label[label <= (self.STATIC[0] + self.eps)] = 0.0
        label[np.where(fg_area)] = 1.0
        label[label > 1.0] = -1.0

        label = np.float32(
            label if self.image_size == None else cv2.resize(label, dsize=self.image_size, interpolation=cv2.INTER_NEAREST)
        )

        return label
