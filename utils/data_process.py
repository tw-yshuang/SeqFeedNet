import os, random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import cv2
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

if __name__ == '__main__':
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import DatasetConfig
from cross_validation_config import datasets_tr, datasets_test
from utils.DataID_MatchTable import VID2ID, CAT2ID
from utils.data_preprocess import CDNet2014Preprocess
from utils.transforms import CustomCompose
from submodules.UsefulFileTools.FileOperator import get_filenames


class CDNet2014OneVideo:
    def __init__(self, cate_name: str, name: str) -> None:
        """
        The above function initializes an object with various attributes and loads paths based on the given
        category name.

        Args:
          cate_name (str): The `cate_name` parameter is a string that represents the category name. It is
        used to specify the category directory where the data is stored.
          name (str): The `name` parameter is a string that represents the name of the object or category.
        """
        self.name = name
        self.id = VID2ID[self.name]

        self.temporalROI: Tuple[int]
        self.gtPaths_all: Tuple[str]
        self.inputPaths_all: Tuple[str]
        self.recentBgPaths_all: Tuple[str]
        self.emptyBgPaths: Tuple[str]

        self.gtPaths_beforeROI = Tuple[str]
        self.inputPaths_beforeROI = Tuple[str]
        self.recentBgPaths_beforeROI = Tuple[str]
        self.gtPaths_inROI = Tuple[str]
        self.inputPaths_inROI = Tuple[str]
        self.recentBgPaths_inROI = Tuple[str]

        with open(file=f'{DatasetConfig.currentFrDir}/{cate_name}/{self.name}/temporalROI.txt', mode='r') as f:
            self.temporalROI = tuple(map(int, f.read().split(' ')))

        self.__load_paths(cate_name)
        self.__split_by_temporalROI()

    def __load_paths(self, cate_name: str):
        for sub_dir, extension, dir_path, var_name in zip(
            ['groundtruth/', 'input/', '', ''],
            ['png', *['jpg'] * 3],
            [DatasetConfig.currentFrDir, DatasetConfig.currentFrDir, DatasetConfig.emptyBgDir, DatasetConfig.recentBgDir],
            ['gtPaths_all', 'inputPaths_all', 'emptyBgPaths', 'recentBgPaths_all'],
        ):
            paths = get_filenames(dir_path=f'{dir_path}/{cate_name}/{self.name}/{sub_dir}', specific_name=f'*.{extension}')
            # exclude Synology NAS snapshot
            setattr(self, var_name, tuple(sorted([path for path in paths if '@eaDir' not in path])))

    def __split_by_temporalROI(self):
        for var_beforeROI, var_Mask, paths in zip(
            ['gtPaths_beforeROI', 'inputPaths_beforeROI', 'recentBgPaths_beforeROI'],
            ['gtPaths_inROI', 'inputPaths_inROI', 'recentBgPaths_inROI'],
            [self.gtPaths_all, self.inputPaths_all, self.recentBgPaths_all],
        ):
            setattr(self, var_beforeROI, (*paths[: self.temporalROI[0]], *paths[self.temporalROI[1] :]))
            setattr(self, var_Mask, paths[self.temporalROI[0] : self.temporalROI[1]])

    def __repr__(self) -> str:
        return self.name


class CDNet2014OneCategory:
    def __init__(self, name: str, ls: List[str]) -> None:
        """
        The function initializes an object with a name, ID, and a list of videos.

        Args:
          name (str): The `name` parameter is a string that represents the name of the object being
        initialized.
          ls (List[str]): The parameter `ls` is a list of strings. Each string represents a video.
        """
        self.name = name
        self.id = CAT2ID[self.name]
        self.videos = [CDNet2014OneVideo(self.name, video_str) for video_str in ls]

    def __repr__(self) -> str:
        return self.name


class CDNet2014Dataset(Dataset):
    GAP = 2
    GAP_ARR: NDArray[np.int16]
    CFG: DatasetConfig

    def __init__(
        self,
        cv_dict: Dict[str, Dict[str, List[str]]] = datasets_tr,
        cv_set: int = 0,
        cfg: DatasetConfig | None = None,
        transforms_cpu: CustomCompose | transforms.Compose = None,
        isShadowFG: bool = False,
        isTrain: bool = True,
    ) -> None:
        """
        This function initializes a dataset object for training or testing purposes, based on the
        provided parameters.

        Args:
          cv_dict (Dict[str, Dict[str, List[str]]]): `cv_dict` is a dictionary that contains
        cross-validation datasets. It has the following structure:
          cv_set (int): The `cv_set` parameter is an integer that represents the cross-validation set to
        use. It is used to select a specific subset of data from the `cv_dict` dictionary. Defaults to 0
          cfg (DatasetConfig | None): The `cfg` parameter is an instance of the `DatasetConfig` class. It
        is used to configure the dataset for training.
          transforms_cpu (CustomCompose | transforms.Compose): The `transforms_cpu` parameter is used to
        specify the transformations to be applied to the data. It can be either an instance of the
        `CustomCompose` class or an instance of the `transforms.Compose` class. These classes are
        typically used in PyTorch for defining a sequence of transformations to be
          isShadowFG (bool): isShadowFG is a boolean flag that indicates whether to include shadow
        foreground in the dataset. Defaults to False
          isTrain (bool): A boolean flag indicating whether the code is being executed in training mode
        or not. Defaults to True
        """
        self.cv_dict = cv_dict[cv_set]  # from cross_validation_config.py
        self.transforms_cpu = transforms_cpu
        self.preprocess = CDNet2014Preprocess(isShadowFG=isShadowFG)
        self.isTrain = isTrain

        self.categories = [CDNet2014OneCategory(name=k, ls=v) for k, v in self.cv_dict.items()]
        self.data_infos: List[Tuple[CDNet2014OneCategory, CDNet2014OneVideo, int]] = []  # [(cate, video, frame_inROI_id)...]

        error_msg = "The type of {} must be {} when is in the {} mode."
        if self.isTrain:
            assert isinstance(cfg, DatasetConfig), error_msg.format('cfg', DatasetConfig, 'training')
            assert isinstance(transforms_cpu, CustomCompose), error_msg.format('transforms_cpu', CustomCompose, 'training')

            CDNet2014Dataset.CFG = cfg
            CDNet2014Dataset.GAP = self.CFG.gap_range[0]
            CDNet2014Dataset.GAP_ARR = np.linspace(*self.CFG.gap_range, self.CFG.num_epoch, dtype=np.int16)
            self.__collect_training_data()
        else:
            assert isinstance(transforms_cpu, transforms.Compose), error_msg.format('transforms_cpu', transforms.Compose, 'testing')
            self.__collect_testing_data()

    def __collect_training_data(self):
        sample4oneVideo = self.CFG.sample4oneVideo

        for cate in self.categories:
            for video in cate.videos:
                idxs = sorted(random.sample(range(len(video.inputPaths_inROI)), k=sample4oneVideo))
                self.data_infos = [*self.data_infos, *list(zip([cate] * sample4oneVideo, [video] * sample4oneVideo, idxs))]

    def __collect_testing_data(self):
        for cate in self.categories:
            for video in cate.videos:
                self.data_infos.append((cate, video, None))

    def __getitem__(self, idx: int) -> Any:
        features: torch.Tensor
        frames: torch.Tensor
        labels: torch.Tensor

        cate, video, frame_id = self.data_infos[idx]
        features = torch.from_numpy(self.__get_features(video).transpose(0, 3, 1, 2)).type(torch.float32) / 255

        if not self.isTrain:
            # ! test_dataset can use this, but test_loader can not use.
            # ! TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'generator'>
            # * video_info, features, genFunc4FrameAndLabel(); genFunc4FrameAndLabel() -> frame, label
            return video.id, self.transforms_cpu(features), self.__getitem4testIter(video)

        frame_ids = self.__get_frameIDs(video, frame_id)
        frame_ls = []
        label_ls = []
        for i in frame_ids:
            frame_ls.append(cv2.imread(video.inputPaths_inROI[i], cv2.COLOR_BGR2RGB))
            label_ls.append(np.expand_dims(self.preprocess(cv2.imread(video.gtPaths_inROI[i], cv2.IMREAD_GRAYSCALE)), axis=-1))

        frames = torch.from_numpy(np.stack(frame_ls).transpose(0, 3, 1, 2)).type(torch.float32) / 255.0
        labels = torch.from_numpy(np.stack(label_ls).transpose(0, 3, 1, 2))

        # * video_info, features, frames, labels
        return video.id, *self.transforms_cpu(frames, labels, features)

    def __getitem4testIter(self, video: CDNet2014OneVideo):
        for input_path, gt_path in zip(video.inputPaths_inROI, video.gtPaths_inROI):
            frame = cv2.imread(input_path, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            frame = torch.from_numpy(np.expand_dims(frame, axis=0)).type(torch.float32) / 255.0
            label = torch.from_numpy(np.expand_dims(self.preprocess(cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)), axis=(0, 1)))
            yield self.transforms_cpu(frame), self.transforms_cpu(label)

    # *dataset selecting strategy
    def __get_frameIDs(self, video: CDNet2014OneVideo, start_id: int) -> List[int]:
        len_frame = len(video.inputPaths_inROI)

        if len_frame - start_id <= self.CFG.frame_groups * self.GAP:
            return sorted(random.sample(range(start_id - self.CFG.frame_groups * self.GAP - 1, len_frame), k=self.CFG.frame_groups))

        frame_ids: List[int] = []
        frame_id = start_id
        for _ in range(self.CFG.frame_groups):
            frame_id += random.randint(1, self.GAP)
            frame_ids.append(frame_id)

        return frame_ids

    def __get_features(self, video: CDNet2014OneVideo, mean=0, std=128):
        f0 = cv2.imread(random.choice(video.emptyBgPaths), cv2.COLOR_BGR2RGB)
        f1 = f0 + np.random.normal(mean, std, f0.shape)

        return np.stack([f0, f1])

    @classmethod
    def next_frame_gap(cls, epoch: int = 1):
        """
        The function updates the value of the "gap" attribute based on the current epoch and the
        "next_stage" value from the configuration.

        Args:
          epoch (int): The epoch parameter represents the current epoch number. Defaults to 1
        """

        cls.GAP = cls.GAP_ARR[epoch]

    def __len__(self):
        return len(self.data_infos)


def get_data_SetAndLoader(
    dataset_cfg: DatasetConfig = DatasetConfig(),
    cv_set: int = 1,
    dataset_rate=1.0,
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = False,
    train_transforms_cpu: CustomCompose = None,
    test_transforms_cpu: transforms.Compose = None,
    label_isShadowFG: bool = False,
    useTestAsVal=False,
    **kwargs,
):
    dataset = CDNet2014Dataset(datasets_tr, cv_set, dataset_cfg, train_transforms_cpu, isShadowFG=label_isShadowFG, isTrain=True)
    train_len = int(len(dataset) * dataset_rate)
    train_set, val_set = random_split(dataset, [train_len, len(dataset) - train_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    val_loader = None

    test_set = CDNet2014Dataset(datasets_test, cv_set, dataset_cfg, test_transforms_cpu, isShadowFG=label_isShadowFG, isTrain=False)
    if useTestAsVal:
        test4val_transforms_cpu = CustomCompose(test_transforms_cpu.transforms)
        val_set = CDNet2014Dataset(
            datasets_test, cv_set, dataset_cfg, test4val_transforms_cpu, isShadowFG=label_isShadowFG, isTrain=True
        )
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    else:
        if dataset_rate != 1.0:
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_set


if __name__ == '__main__':
    from tqdm import tqdm
    from utils.transforms import RandomCrop, RandomResizedCrop

    sizeHW = (224, 224)
    trans = CustomCompose(
        [
            # transforms.ToTensor(), # already converted in the __getitem__()
            transforms.RandomChoice(
                [
                    RandomCrop(crop_size=sizeHW, p=1.0),
                    RandomResizedCrop(sizeHW, scale=(0.6, 1.6), ratio=(3.0 / 5.0, 2.0), p=0.9),
                ]
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # # ! need pop error
    # trans = transforms.Compose(
    #     [
    #         transforms.RandomCrop(size=(224, 224)),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )

    dataset = CDNet2014Dataset(cv_dict=datasets_tr, cv_set=5, cfg=DatasetConfig(), transforms_cpu=trans)

    print(dataset.categories[0].name)
    print(dataset.categories[0].videos[0].name)
    print(dataset.categories[0].videos[0].gtPaths_beforeROI[0])

    print(dataset.data_infos)
    print(len(dataset.data_infos))

    info, features, frames, labels = next(iter(dataset))
    print(info)
    print(frames)
    print(features)
    print(labels)

    # =============================

    test_trans = transforms.Compose(
        [
            transforms.Resize(size=(244, 244)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # # ! need pop error
    # trans = CustomCompose(
    #     [
    #         transforms.RandomCrop(size=(224, 224)),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )

    data_test = CDNet2014Dataset(cv_dict=datasets_test, cv_set=5, transforms_cpu=test_trans, isTrain=False)
    info, features, iterFandL = next(iter(data_test))

    print(info)
    print(features)
    for i, (frame, label) in enumerate(iterFandL):
        print(i)

    # =============================

    cfg = DatasetConfig()
    train_trans_cpu = CustomCompose(
        [
            # transforms.ToTensor(), # already converted in the __getitem__()
            transforms.RandomChoice(
                [
                    RandomCrop(crop_size=sizeHW, p=1.0),
                    RandomResizedCrop(sizeHW, scale=(0.6, 1.6), ratio=(3.0 / 5.0, 2.0), p=0.9),
                ]
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_trans_cpu = transforms.Compose(
        [
            # transforms.ToTensor(), # already converted in the __getitem__()
            transforms.Resize(sizeHW, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset_cfg = DatasetConfig()
    dataset_cfg.next_stage = 1  # ? test used

    dataset, test_set, train_loader, val_loader = get_data_SetAndLoader(
        dataset_cfg=dataset_cfg,
        cv_set=5,
        dataset_rate=1,
        batch_size=8,
        num_workers=16,
        pin_memory=True,
        train_transforms_cpu=train_trans_cpu,
        test_transforms_cpu=test_trans_cpu,
        label_isShadowFG=False,
        useTestAsVal=True,
    )

    i = 0
    for video_info, features, frames, labels in tqdm(train_loader):
        if i == 0:
            i += 1
            print(video_info)
            print(features)
            print(frames)
            print(labels)

    test_set = CDNet2014Dataset(datasets_test, 5, None, test_trans, isShadowFG=False, isTrain=False)
    for i, (video_info, features, iterFandL) in enumerate(test_set):
        print(f"test: {i}")
        if i == 0:
            print(video_info)
            print(features)
        for j, (frame, label) in enumerate(iterFandL):
            print(f"iter: {j}")
            if j == 0:
                print(frame)
                print(label)
