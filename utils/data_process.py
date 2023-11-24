import os, random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from submodules.UsefulFileTools.FileOperator import get_filenames
from cross_validation_config import datasets_tr, datasets_test


class CDNet2014OneCategory:
    currentFrDir = 'Data/currentFr'
    emptyBgDir = 'Data/emptyBg'
    recentBgDir = 'Data/recentBg'

    def __init__(self, name: str, ls: List[str]) -> None:
        self.name = name
        self.ls = ls

        self.groundtruth_paths: list[str] = []
        self.input_paths: list[str] = []
        self.emptyBg_paths: list[str] = []
        self.recentBg_paths: list[str] = []

        for video_str in self.ls:
            for sub_dir, extension, dir_path, cate_paths in zip(
                ['groundtruth/', 'input/', '', ''],
                ['png', *['jpg'] * 3],
                [self.currentFrDir, self.currentFrDir, self.emptyBgDir, self.recentBgDir],
                [self.groundtruth_paths, self.input_paths, self.emptyBg_paths, self.recentBg_paths],
            ):
                cate_paths.append(
                    get_filenames(dir_path=f'{dir_path}/{self.name}/{video_str}/{sub_dir}', specific_name=f'*.{extension}')
                )
                cate_paths[-1] = sorted([path for path in cate_paths[-1] if '@eaDir' not in path])  # exclude Synology NAS snapshot


class CDNet2014Dataset:
    badWeather: CDNet2014OneCategory
    baseline: CDNet2014OneCategory
    cameraJitter: CDNet2014OneCategory
    dynamicBackground: CDNet2014OneCategory
    intermittentObjectMotion: CDNet2014OneCategory
    lowFramerate: CDNet2014OneCategory
    nightVideos: CDNet2014OneCategory
    PTZ: CDNet2014OneCategory
    shadow: CDNet2014OneCategory
    thermal: CDNet2014OneCategory
    turbulence: CDNet2014OneCategory

    def __init__(self, dataset_category: Dict[str, List[str]] = datasets_tr, cross_set: int = 0) -> None:
        self.dataset_category = dataset_category[cross_set]  # from cross_validation_config.py

        for k, v in self.dataset_category.items():
            setattr(self, k, CDNet2014OneCategory(name=k, ls=v))


if __name__ == '__main__':
    dataset = CDNet2014Dataset(dataset_category=datasets_tr, cross_set=5)
    print(dataset.baseline.emptyBg_paths)
    print(len(dataset.baseline.emptyBg_paths))
