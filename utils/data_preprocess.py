from typing import Tuple

import numpy as np
import cv2
import torch


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

    def __call__(self, gt: np.ndarray) -> np.ndarray:
        label = np.float32(gt if self.image_size == None else cv2.resize(gt, dsize=self.image_size))

        if self.isShadowFG:
            label[np.where((label <= self.PXL_VAL_SHADOW + self.eps) & (label >= self.PXL_VAL_SHADOW - self.eps))] = 1.0

        label[label < self.PXL_VAL_SHADOW - self.eps] = 0.0
        label[label > self.PXL_VAL_MOVING - self.eps] = 1.0
        label[label > 1.0] = -1.0

        return label


# TODO
class LASIESTA_preprocess:
    # * Define gt pixel value meaning
    MOVING2STATIC = (255, 255, 255)
    STATIC = (0, 0, 0)
    UNKNOWN = (128, 128, 128)  # Belong to NON roi
    OBJ1 = (255, 0, 0)
    OBJ2 = (0, 255, 0)
    OBJ3 = (255, 255, 0)

    @classmethod
    def preprocessing(cls, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gt_preprocessed = torch.zeros_like(gt)
        gt_preprocessed = -1

        return gt_preprocessed


if __name__ == '__main__':
    import sys
    from pathlib import Path

    PROJECT_DIR = Path(__file__).resolve().parents[1]
    sys.path.append(str(PROJECT_DIR))

    from submodules.UsefulFileTools.FileOperator import get_filenames

    filenames = sorted(get_filenames('Data/CDNet2014/**/groundtruth', '*.png'))
    print(len(filenames))

    # ! Test for CDNet2014
    data_processes = CDNet2014Preprocess(image_size=(244, 244), isShadowFG=False, eps=5)  # all pass
    for i, filename in enumerate([filenames[9842], filenames[26842], filenames[65697], filenames[115181], filenames[125681]]):
        print(filename)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        print(img.shape, end=' ')
        convert_img = data_processes(img)
        print(np.any(convert_img == 0.0), np.any(convert_img == 1.0), np.any(convert_img == -1.0))
        convert_img[convert_img == -1] = 127
        convert_img[convert_img == 1] = 255
        cv2.imwrite(f'out/test/original_{i}.png', img)
        cv2.imwrite(f'out/test/convert_{i}.png', convert_img)
