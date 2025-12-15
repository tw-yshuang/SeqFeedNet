import os, random
from pathlib import Path

# from typing import Any, Dict, List, Tuple

import numpy as np
import cv2
from numpy.typing import NDArray

if __name__ == '__main__':
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from submodules.UsefulFileTools.FileOperator import get_filenames
from submodules.UsefulFileTools.WordOperator import str_format


if __name__ == '__main__':
    PXL_VAL_STATIC = 0
    PXL_VAL_SHADOW = 50
    PXL_VAL_NONROI = 85
    PXL_VAL_UNKNOWN = 170
    PXL_VAL_MOVING = 255

    source_dir = './Data/currentFr/'
    target_dir = './Data/moveObjCrop/'
    suffix_input_dir = 'input/in'
    suffix_gt_dir = 'groundtruth/gt'
    input_extension = 'jpg'
    gt_extension = 'png'

    len_source_dir = len(source_dir)
    filenames = sorted(get_filenames(f'{source_dir}', specific_name=f'{suffix_input_dir}*.{input_extension}'))

    cate_dir_dict = {}

    # print(filenames)

    for filename in filenames:
        img = cv2.imread(filename)
        mask = cv2.imread(filename.replace(suffix_input_dir, suffix_gt_dir)[:-3] + gt_extension, cv2.IMREAD_GRAYSCALE)
        cate_dir, filename = f'{target_dir}{filename[len_source_dir:]}'.split(suffix_input_dir)
        # print(cate_dir, filename)

        moving_mask = np.zeros_like(mask)
        moving_mask[mask >= PXL_VAL_UNKNOWN] = 1  # UNKNOWN + MOVING part
        # cv2.morphologyEx(moving_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype='uint8'), iterations=1)
        contours, hierarchy = cv2.findContours(moving_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        moving_mask = moving_mask[:, :, np.newaxis]
        # moving_img = img * moving_mask[:, :, np.newaxis]
        i = 0
        for cont in contours:
            bbox = cv2.boundingRect(cont)
            x, x_end, y, y_end = bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]
            orig_img = img[x:x_end, y:y_end]
            orig_mask = mask[x:x_end, y:y_end]
            target_mask = moving_mask[x:x_end, y:y_end]

            # * filter out the noise moving object
            if (
                bbox[2] / bbox[3] < 1  # wide shape object
                or target_mask.size < 30**2  # minimum size
                or target_mask.sum() / target_mask.size < 0.5  # the minimum size of object in the bbox
                or orig_mask[orig_mask == PXL_VAL_MOVING].size / target_mask.size < 0.3  # the minimum size of moving part in the bbox
            ):
                continue

            # after filter, add the shadow to avoid too much shadow data
            target_mask[orig_mask == PXL_VAL_SHADOW] = 1
            target_img = orig_img * target_mask

            if not Path.exists(Path(cate_dir)):
                os.makedirs(cate_dir, mode=0o755, exist_ok=True)
                print(str_format(f"Create directory: {cate_dir} !!", fore='y'))

            cate_dir_dict[cate_dir] = cate_dir_dict.setdefault(cate_dir, 0) + 1

            i += 1
            path = f'{cate_dir}{filename[:-4]}-{i:02d}.png'
            cv2.imwrite(path, target_img)
            print(str_format(f"Write image: {path} !!", fore='g'))

    for k, v in cate_dir_dict.items():
        print(f"{k}: {v}")
