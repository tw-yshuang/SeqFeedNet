import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import cv2
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.io import read_image

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

        self.ROI_mask: torch.Tensor
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

        video_dir = f'{DatasetConfig.currentFrDir}/{cate_name}/{self.name}'
        self.ROI_mask = torch.from_numpy(cv2.imread(f'{video_dir}/ROI.bmp', cv2.IMREAD_GRAYSCALE)) != 0
        with open(file=f'{video_dir}/temporalROI.txt', mode='r') as f:
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
    GAP = 3
    GAP_ARR: NDArray[np.int16]
    CFG: DatasetConfig
    isVal: bool

    def __init__(
        self,
        cv_dict: Dict[str, Dict[str, List[str]]] = datasets_tr,
        cv_set: int = 0,
        cfg: DatasetConfig | None = None,
        transforms_cpu: CustomCompose | transforms.Compose = None,
        isShadowFG: bool = False,
        isTrain: bool = True,
        isVal: bool = False,
        testFromBegin: bool = True,
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
        self.isVal = isVal
        self.testFromBegin = testFromBegin

        self.categories = [CDNet2014OneCategory(name=k, ls=v) for k, v in self.cv_dict.items()]
        self.data_infos: List[Tuple[CDNet2014OneCategory, CDNet2014OneVideo, int]] = []  # [(cate, video, frame_inROI_id)...]

        error_msg = "The type of {} must be {} when is in the {} mode."
        assert isinstance(cfg, DatasetConfig), error_msg.format('cfg', DatasetConfig, 'training')
        CDNet2014Dataset.CFG = cfg

        if self.isTrain:
            assert isinstance(transforms_cpu, CustomCompose), error_msg.format('transforms_cpu', CustomCompose, 'training')

            CDNet2014Dataset.GAP = self.CFG.gap_range[0]
            CDNet2014Dataset.GAP_ARR = np.linspace(*self.CFG.gap_range, self.CFG.num_epochs, dtype=np.int16)
            self.__collect_training_data()
        else:
            assert isinstance(transforms_cpu, transforms.Compose), error_msg.format('transforms_cpu', transforms.Compose, 'testing')
            self.__collect_testing_data()

    def __collect_training_data(self):
        sample4oneVideo = self.CFG.sample4oneVideo

        for cate in self.categories:
            for video in cate.videos:
                # idxs = sorted(random.sample(range(len(video.inputPaths_inROI)), k=sample4oneVideo))
                idxs = list(map(lambda x: x - video.temporalROI[0], self.CFG.cdnet2014SelectedMap[video.id]))
                self.data_infos = [*self.data_infos, *list(zip([cate] * sample4oneVideo, [video] * len(idxs), idxs))]

    def __collect_testing_data(self):
        for cate in self.categories:
            for video in cate.videos:
                self.data_infos.append((cate, video, 0))

    @staticmethod
    def __read_empty(empty_path: str):
        return read_image(empty_path).type(torch.float32) / 255.0

    @staticmethod
    def __median_empty(empty_paths: Tuple[str]):
        empty_median = torch.median(torch.stack([read_image(empty_path) for empty_path in empty_paths], dim=0), dim=0)[0]
        empty_median = empty_median.type(torch.float32) / 255.0

        def __return_empty_median(*args):
            return empty_median

        return __return_empty_median

    def __getitem__(self, idx: int) -> Any:
        features: torch.Tensor
        frames: torch.Tensor
        empty_frames: torch.Tensor
        labels: torch.Tensor

        cate, video, frame_id = self.data_infos[idx]
        frame_ids, gap = self.__get_frameIDs(video, cate.id, frame_id)

        get_empty: Callable
        if cate.id != CAT2ID['PTZ']:
            # emptyBg4InputPaths = tuple(random.choices(video.emptyBgPaths, k=self.CFG.frame_groups))
            emptyBg4InputPaths = tuple(random.choices(video.emptyBgPaths, k=30))
            get_empty = self.__median_empty(emptyBg4InputPaths)
        else:
            len_beforeROI = len(video.inputPaths_beforeROI)
            emptyBg4InputPaths = tuple([video.emptyBgPaths[len_beforeROI + f_id] for f_id in frame_ids])
            get_empty = self.__read_empty

        if not self.isTrain:
            # ! test_dataset can use this, but test_loader can not use.
            # ! TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'generator'>
            features = self.__get_features(video, cate.id, frame_id, get_empty)
            # * video_info, features, genFunc4FrameAndLabel(); genFunc4FrameAndLabel() -> frame, label
            return video.id, self.transforms_cpu(features), self.__getitem4testIter(video, cate.id)

        features = self.__get_features(video, cate.id, frame_ids[0], get_empty)

        frame_ls = []
        empty_ls = []
        label_ls = []
        for i, f_id in enumerate(frame_ids):
            frame_ls.append(read_image(video.inputPaths_inROI[f_id]))
            # empty_ls.append(read_image(emptyBg4InputPaths[i]))
            empty_ls.append(features[0])
            label_ls.append(self.preprocess(read_image(video.gtPaths_inROI[f_id])))

        frames = torch.stack(frame_ls).type(torch.float32) / 255.0
        empty_frames = torch.stack(empty_ls)
        # features[-1] = torch.abs(features[1] - frames[0])
        labels = torch.stack(label_ls)

        # * video_info, frames, empty_frames, labels, features
        return video.id, *self.transforms_cpu(frames, empty_frames, labels, features, video.ROI_mask, video.id, gap)

    def __getitem4testIter(self, video: CDNet2014OneVideo, cate_id: int):
        if self.testFromBegin:
            inputPaths = video.inputPaths_all
            gtPaths = video.gtPaths_all
            if cate_id != CAT2ID['PTZ']:
                emptyBg4InputPaths = tuple(sorted(random.choices(video.emptyBgPaths, k=len(video.inputPaths_all))))
            else:
                emptyBg4InputPaths = video.emptyBgPaths
        else:
            inputPaths = video.inputPaths_inROI
            gtPaths = video.gtPaths_inROI
            if cate_id != CAT2ID['PTZ']:
                emptyBg4InputPaths = tuple(random.choices(video.emptyBgPaths, k=len(video.inputPaths_inROI)))
            else:
                emptyBg4InputPaths = video.emptyBgPaths[len(video.inputPaths_beforeROI) :]

        get_empty: Callable
        if cate_id != CAT2ID['PTZ']:
            get_empty = self.__median_empty(tuple(sorted(random.choices(video.emptyBgPaths, k=30))))
        else:
            get_empty = self.__read_empty

        for input_path, empty_path, gt_path in zip(inputPaths, emptyBg4InputPaths, gtPaths):
            frame = read_image(input_path).unsqueeze(0).type(torch.float32) / 255.0
            empty = get_empty(empty_path).unsqueeze(0)
            label = self.preprocess(read_image(gt_path)).unsqueeze(0)
            yield self.transforms_cpu(frame), self.transforms_cpu(empty), self.transforms_cpu(label)

    # *dataset selecting strategy
    def __get_frameIDs(self, video: CDNet2014OneVideo, cate_id: int, selected_id: int) -> Tuple[List[int], int]:
        len_frame = len(video.inputPaths_inROI)
        selectedID_pos = random.randint(0, self.CFG.frame_groups - 1)

        if self.isVal or cate_id == CAT2ID['lowFramerate']:
            start_idx = selected_id - selectedID_pos
            end_idx = start_idx + self.CFG.frame_groups
            if 0 > start_idx:
                start_idx = 0
                end_idx = self.CFG.frame_groups
            elif len_frame < end_idx:
                start_idx = len_frame - self.CFG.frame_groups
                end_idx = len_frame

            return list(range(start_idx, end_idx)), 1

        while True:
            # normally, the PTZPan effect has less moving
            gap = random.randint(1, max(5, self.GAP // 10) if cate_id == CAT2ID['PTZ'] else self.GAP)

            if gap * self.CFG.frame_groups > len_frame:
                continue

            candidateID_first = selected_id - gap * selectedID_pos
            candidateID_last = candidateID_first + gap * self.CFG.frame_groups

            if 0 > candidateID_first:
                candidateID_first = 0
            elif len_frame <= candidateID_last:
                candidateID_first = len_frame - self.CFG.frame_groups * gap

            frame_id = random.randint(candidateID_first, candidateID_first + gap - 1)
            frame_ids = [frame_id + i * gap for i in range(self.CFG.frame_groups)]

            return frame_ids, gap

    def __get_features(self, video: CDNet2014OneVideo, cate_id: int, frame_id: int, get_empty: Callable):

        # if cate_id == CAT2ID['PTZ']:
        #     f0 = read_image(video.emptyBgPaths[len(video.inputPaths_beforeROI) + frame_id])
        # else:
        #     f0 = read_image(random.choice(video.emptyBgPaths))

        f0 = get_empty(video.emptyBgPaths[len(video.inputPaths_beforeROI) + frame_id] if cate_id == CAT2ID['PTZ'] else None)
        f1 = read_image(video.recentBgPaths_inROI[frame_id]).type(torch.float32) / 255.0
        f2 = read_image(video.inputPaths_inROI[frame_id]).type(torch.float32) / 255.0

        return torch.stack([f0, f1, torch.abs(f1 - f2)])

    @classmethod
    def update_frame_gap(cls, epoch: int = 1):
        """
        The function updates the value of the "gap" attribute based on the current epoch and the
        "next_stage" value from the configuration.

        Args:
          epoch (int): The epoch parameter represents the current epoch number. Defaults to 1
        """

        cls.GAP = cls.GAP_ARR[epoch]

    def __len__(self):
        return len(self.data_infos)


def get_data_LoadersAndSet(
    dataset_cfg: DatasetConfig = DatasetConfig(),
    cv_set: int = 1,
    dataset_rate=1.0,
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = False,
    train_transforms_cpu: CustomCompose = None,
    test_transforms_cpu: transforms.Compose = None,
    label_isShadowFG: bool = False,
    useTestAsVal: bool = False,
    onlyTest: bool = False,
    testFromBegin: bool = True,
    **kwargs,
):
    test_set = CDNet2014Dataset(
        datasets_test,
        cv_set,
        dataset_cfg,
        test_transforms_cpu,
        isShadowFG=label_isShadowFG,
        isTrain=False,
        testFromBegin=testFromBegin,
    )

    if onlyTest:
        return None, None, test_set

    dataset = CDNet2014Dataset(datasets_tr, cv_set, dataset_cfg, train_transforms_cpu, isShadowFG=label_isShadowFG, isTrain=True)
    train_len = int(len(dataset) * dataset_rate)
    train_set, val_set = random_split(dataset, [train_len, len(dataset) - train_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    val_loader = None

    if useTestAsVal:
        test4val_transforms_cpu = CustomCompose(test_transforms_cpu.transforms)
        val_set = CDNet2014Dataset(
            datasets_test, cv_set, dataset_cfg, test4val_transforms_cpu, isShadowFG=label_isShadowFG, isTrain=True, isVal=True
        )
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    elif dataset_rate != 1.0:
        # val_set.isVal = True # TODO: not sure
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_set
