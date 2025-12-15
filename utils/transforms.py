import math, random
from copy import deepcopy
from typing import Dict, List, Callable, Tuple

import torch
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

if __name__ == '__main__':
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import DatasetConfig
from cross_validation_config import datasets_tr
from utils.DataID_MatchTable import ID2CAT, ID2VID, CAT2ID, get_VID2CID
from submodules.UsefulFileTools.FileOperator import get_filenames


MINIMUM_ROI_AREA_RATE = 0.05 * 0.05


def monkeyPatch4Compose__call__(self, img: torch.Tensor):
    for t in self.transforms:
        if img.shape[-3] == 1:
            match t.__class__.__name__:
                case transforms.Normalize.__name__:
                    continue
                case transforms.Resize.__name__:
                    img = TF.resize(img, t.size, InterpolationMode.NEAREST, t.max_size)
        img = t(img)
    return img


# transforms.RandomChoice only for training used
def monkeyPatch4Choice__call__(self, *args):
    t = random.choices(self.transforms, weights=self.p)[0]
    return t


transforms.Compose.__call__ = monkeyPatch4Compose__call__
transforms.RandomChoice.__call__ = monkeyPatch4Choice__call__


class CustomCompose:
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(
        self,
        frames: torch.Tensor,
        empty_frames: torch.Tensor,
        labels: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        roi_mask: torch.Tensor | None = None,
        video_id: int | None = None,
        gap: int | None = None,
    ):
        for t in self.transforms:
            if t.__module__ != transforms.transforms.__name__:
                # print(t)
                frames, empty_frames, labels, features, roi_mask = t(frames, empty_frames, labels, features, roi_mask=roi_mask)
            else:
                if t.__class__.__name__ == AdditiveColorJitter.__name__:
                    print(torch.any(frames < 0))

                if t.__class__.__name__ == transforms.RandomChoice.__name__:
                    t = t()
                    if t.__module__ != transforms.transforms.__name__:
                        # print(t)
                        frames, empty_frames, labels, features, roi_mask = t(
                            frames, empty_frames, labels, features, roi_mask=roi_mask, video_id=video_id, gap=gap
                        )
                        continue

                # print(t)
                if features is not None:
                    features = t(features)
                if any(keyword in t.__class__.__name__ for keyword in ['Crop', 'Resize', 'Flip']):
                    if 'Random' in t.__class__.__name__:
                        raise RuntimeError(
                            "Do not use space-related random-augmentation methods from transforms.transforms, need to keep space domain related for three outputs"
                        )
                    # * labels do not use the augmentation method from transforms.transforms, labels is binary-like info so it do not use color-related augmentation
                    labels = t(labels)

                frames = t(frames)
                empty_frames = t(empty_frames)

        return frames, empty_frames, labels, features

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class IterativeCustomCompose:
    def __init__(self, transforms: List[Callable], target_size=(224, 224), device: str = 'cuda') -> None:
        '''
        target_size: (H, W)
        '''
        self.compose = CustomCompose(transforms)
        self.target_size = target_size
        self.device = device

        self.parameter_order = ['frames', 'empty_frames', 'labels', 'features']

    def __call__(
        self,
        b_frames: torch.Tensor,
        b_empty_frames: torch.Tensor,
        b_labels: torch.Tensor | None = None,
        b_features: torch.Tensor | None = None,
        useBuffer: bool = False,
    ):
        b_dict: Dict[str, torch.Tensor] = {}
        for b_items, name in zip([b_frames, b_empty_frames, b_labels, b_features], self.parameter_order):
            if b_items is not None:
                b_dict[name] = b_items

        if useBuffer:
            process_b_dict = {
                k: torch.zeros((*v.shape[0:3], *self.target_size), dtype=torch.float32).to(self.device) for k, v in b_dict.items()
            }
        else:
            process_b_dict = b_dict

        for i, items_tuple in enumerate(zip(*process_b_dict.values())):
            items_dict = {k: v for k, v in zip(process_b_dict.keys(), items_tuple)}
            output_tuple = self.compose(**items_dict)
            for output, name in zip(output_tuple, self.parameter_order):
                if process_b_dict.setdefault(name, None) is not None:
                    process_b_dict[name][i] = output

        assert (
            process_b_dict['frames'].shape[-2:] == self.target_size
        ), f"The size of input parameters are not same with target_size {self.target_size} please use useBuffer=True"
        return process_b_dict['frames'], process_b_dict['empty_frames'], process_b_dict['labels'], process_b_dict['features']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


def get_tlhw_cropIncludeROI(
    minimum_roi_area: int,
    roi_mask: torch.Tensor,
    *args,
    CropMethod: transforms.RandomCrop | transforms.RandomResizedCrop = transforms.RandomCrop,
):
    while True:
        t, l, h, w = CropMethod.get_params(roi_mask, *args)
        if torch.where(TF.F_t.crop(roi_mask, t, l, h, w) == 1)[0].shape[0] > minimum_roi_area:
            break
    return t, l, h, w


# def CheckSize(sizeHW=(224, 224)):
#     def __CheckSize(*inputs, roi_mask: torch.Tensor | None = None):
#         assert inputs[0].shape[-2:]


def RandomCrop(crop_size=(224, 224), p=0.5):
    minimum_roi_area = int(crop_size[0] * crop_size[1] * MINIMUM_ROI_AREA_RATE)

    def __RandomCrop(*inputs, roi_mask: torch.Tensor | None = None, **kwargs):
        if random.random() < p:
            if roi_mask is None:
                t, l, h, w = transforms.RandomCrop.get_params(inputs[0], crop_size)
            else:
                t, l, h, w = get_tlhw_cropIncludeROI(minimum_roi_area, roi_mask, crop_size)

            inputs = [TF.F_t.crop(input, t, l, h, w) for input in inputs]

        return *inputs, roi_mask

    return __RandomCrop


def RandomHorizontalFlip(p=0.5):
    def __RandomHorizontalFlip(*inputs, roi_mask: torch.Tensor | None = None, **kwargs):
        if random.random() < p:
            inputs = [TF.F_t.hflip(input) for input in inputs]

        return *inputs, roi_mask

    return __RandomHorizontalFlip


def RandomVerticalFlip(p=0.5):
    def __RandomVerticalFlip(*inputs, roi_mask: torch.Tensor | None = None, **kwargs):
        if random.random() < p:
            inputs = [TF.F_t.vflip(input) for input in inputs]

        return *inputs, roi_mask

    return __RandomVerticalFlip


def RandomResizedCrop(
    sizeHW: List[int] = (224, 224),
    scale: List[float] = (0.6, 1.6),
    ratio: List[float] = (3.0 / 5.0, 2.0 / 1.0),
    p: float = 0.5,
):
    minimum_roi_area = int(sizeHW[0] * sizeHW[1] * MINIMUM_ROI_AREA_RATE)

    def __RandomResizedCrop(
        frames: torch.Tensor,
        empty_frames: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        roi_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        if random.random() < p:
            if roi_mask is None:
                t, l, h, w = transforms.RandomResizedCrop.get_params(frames, scale, ratio)
            else:
                t, l, h, w = get_tlhw_cropIncludeROI(minimum_roi_area, roi_mask, scale, ratio, CropMethod=transforms.RandomResizedCrop)

            features = TF.F_t.crop(features, t, l, h, w)
            frames = TF.F_t.crop(frames, t, l, h, w)
            empty_frames = TF.F_t.crop(empty_frames, t, l, h, w)
            labels = TF.F_t.crop(labels, t, l, h, w)

        features = TF.F_t.resize(features, size=sizeHW, antialias=True)
        frames = TF.F_t.resize(frames, size=sizeHW, antialias=True)
        empty_frames = TF.F_t.resize(empty_frames, size=sizeHW, antialias=True)
        labels = TF.resize(labels, size=sizeHW, interpolation=InterpolationMode.NEAREST)

        return frames, empty_frames, labels, features, roi_mask

    return __RandomResizedCrop


def RandomShiftedCrop(crop_size: Tuple[int] = (224, 224), max_shift: int = 5, p: float = 0.9):
    minimum_roi_area = int(crop_size[0] * crop_size[1] * MINIMUM_ROI_AREA_RATE)
    randomCrop = RandomCrop(crop_size, p=1.0)

    def __RandomShiftedCrop(
        frames: torch.Tensor,
        empty_frames: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        roi_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        if random.random() > p:
            return randomCrop(frames, empty_frames, labels, features, roi_mask=roi_mask)

        if roi_mask is None:
            t, l, h, w = transforms.RandomCrop.get_params(frames, crop_size)
        else:
            t, l, h, w = get_tlhw_cropIncludeROI(minimum_roi_area, roi_mask, crop_size)

        empty_shift_h, empty_shift_w, recent_shift_h, recent_shift_w = random.choices(list(range(-max_shift, max_shift + 1)), k=4)

        frames = TF.F_t.crop(frames, t, l, h, w)
        labels = TF.F_t.crop(labels, t, l, h, w)
        empty_frames = TF.F_t.crop(empty_frames, t + empty_shift_h, l + empty_shift_w, h, w)

        # * features: ERD -> empty, recent, recent - firstCurrent
        new_features = torch.zeros((*features.shape[:2], *crop_size), dtype=frames.dtype)
        new_features[0] = empty_frames[0]
        new_features[1] = TF.F_t.crop(features[1], t + recent_shift_h, l + recent_shift_w, h, w)
        new_features[2] = torch.abs(new_features[1] - frames[0])

        return frames, empty_frames, labels, new_features, roi_mask

    return __RandomShiftedCrop


class StaticObjSelector:
    def __init__(
        self, cv_set: int, cfg: DatasetConfig, scale_rate: Tuple[float, float] = (0.75, 1.5), max_objScale2Frame: float = 0.35
    ) -> None:
        assert isinstance(cv_set, int), "The type of cv_set must be int!!"
        assert isinstance(cfg, DatasetConfig), "The type of cfg must be DatasetConfig!!"

        self.staticObj_dict = self.get_staticObjDict(cv_set, cfg)
        self.scale_rate = scale_rate
        self.max_scale2frame = max_objScale2Frame

    @staticmethod
    def get_staticObjDict(cv_set: int, cfg: DatasetConfig) -> dict[str, dict[str, list[str]]]:
        cate_dict = datasets_tr[cv_set]
        staticObj_dict = dict(zip(cate_dict.keys(), [{}] * len(cate_dict.keys())))

        for cate_name, video_names in cate_dict.items():
            staticObj_dict[cate_name] = {}
            for v_name in video_names:
                filenames = get_filenames(dir_path=f'{cfg.moveObjCropDir}/{cate_name}/{v_name}', specific_name='*.png')
                if len(filenames) != 0:
                    staticObj_dict[cate_name][v_name] = [f for f in filenames if '@eaDir' not in f]  # exclude Synology NAS snapshot

            if staticObj_dict[cate_name] == {}:
                staticObj_dict.pop(cate_name)

        return staticObj_dict

    def get_staticObj_path(self, target_video_id: int):
        cate_name = ID2CAT[get_VID2CID(target_video_id)]
        video_name = ID2VID[target_video_id]
        if cate_name not in self.staticObj_dict.keys():
            return ''
        elif video_name in self.staticObj_dict[cate_name].keys():
            return random.choice(self.staticObj_dict[cate_name][video_name])
        else:
            return random.choice(self.staticObj_dict[cate_name][random.choice(list(self.staticObj_dict[cate_name].keys()))])

    def insert_staticObj2sample(
        self,
        staticObj_img: torch.Tensor,
        frames: torch.Tensor,
        labels: torch.Tensor,
        empty_frames: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        isCover: bool = False,
        isIntermittent: bool = False,
    ):

        scale_rate = torch.tensor(self.scale_rate, dtype=torch.float32)
        h, w = staticObj_img.shape[-2:]
        max_h, max_w = frames.shape[-2] * self.max_scale2frame, frames.shape[-1] * self.max_scale2frame

        getSmaller = False
        if w > max_w or h > max_h:
            getSmaller = True
            zoomIn_x, zoomIn_y = max_w / w, max_h / h
            scale_rate *= zoomIn_x if zoomIn_x < zoomIn_y else zoomIn_y

        elif w * scale_rate[-1] > max_w or h * scale_rate[-1] > max_h:
            zoomOut_x, zoomOut_y = max_w / w, max_h / h
            scale_rate[-1] = zoomOut_x if zoomOut_x < zoomOut_y else zoomOut_y

        resizeHW = (torch.tensor(staticObj_img.shape[-2:]) * random.uniform(*scale_rate)).type(torch.int)
        if getSmaller or random.random() > 0.25:
            staticObj_img = TF.resize(
                staticObj_img,
                size=resizeHW,
                interpolation=InterpolationMode.NEAREST,
            )
            h, w = resizeHW

        begin_maximum_x, begin_maximum_y = frames.shape[-1] - w, frames.shape[-2] - h
        begin_x, begin_y = random.randint(0, begin_maximum_x), random.randint(0, begin_maximum_y)
        end_x, end_y = begin_x + w, begin_y + h
        staticObj_pixel = staticObj_img > 0.0

        if empty_frames is not None and not isIntermittent:
            empty_frames[..., begin_y:end_y, begin_x:end_x][..., staticObj_pixel] = staticObj_img[staticObj_pixel]

        if isCover:
            crop_labels = labels[..., begin_y:end_y, begin_x:end_x]
            novalid_area = crop_labels == -1
            crop_labels[..., staticObj_img.sum(dim=0) > 0] = 0.0
            crop_labels[novalid_area] = -1

        targetObj_frames = frames * labels
        targetObj_pixel = targetObj_frames > 0.0

        frames[..., begin_y:end_y, begin_x:end_x][..., staticObj_pixel] = staticObj_img[staticObj_pixel]
        frames[targetObj_pixel] = targetObj_frames[targetObj_pixel]

        if features is not None:
            if isIntermittent:
                labels[..., begin_y:end_y, begin_x:end_x][..., staticObj_pixel.any(dim=0)] = 1.0
            else:
                features[1, ..., begin_y:end_y, begin_x:end_x][..., staticObj_pixel] = staticObj_img[staticObj_pixel]
            features[0] = empty_frames[0]
            features[-1] = torch.abs(features[1] - frames[0])

        return frames, labels, empty_frames, features

    def apply(
        self,
        frames: torch.Tensor,
        labels: torch.Tensor,
        empty_frames: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
        target_video_id: int = None,
    ):
        staticObj_path = self.get_staticObj_path(target_video_id)

        if staticObj_path == '':
            return frames, labels, empty_frames, features

        staticObj_img = read_image(staticObj_path).type(torch.float32) / 255.0
        return self.insert_staticObj2sample(
            staticObj_img, frames, labels, empty_frames, features, isCover=random.choice([True, False]), isIntermittent=False
        )

    def duplicate_apply(
        self,
        frames: torch.Tensor,
        empty_frames: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        target_video_id: int,
        objCoverInfos: List[bool] = [False],
        IntermittentObjInfos: List[bool] = [False],
    ):
        staticObj_path = self.get_staticObj_path(target_video_id)
        if staticObj_path != '':
            for isCover, isIntermittent in zip(objCoverInfos, IntermittentObjInfos):
                staticObj_img = read_image(staticObj_path).type(torch.float32) / 255.0
                frames, labels, empty_frames, features = self.insert_staticObj2sample(
                    staticObj_img, frames, labels, empty_frames, features, isCover=isCover, isIntermittent=isIntermittent
                )

                staticObj_path = self.get_staticObj_path(target_video_id)

        return frames, labels, empty_frames, features


def PTZPanCrop(
    crop_size: Tuple[int] = (224, 224),
    empty_move_time: int = 10,
    recent_move_time: int = 5,
    p4targets: float = 0.75,
    p4others: float = 0.9,
    p4extraStaticObj: float = 0.0,
    extraStaticObj_CV_SET: None | int = None,
    datasetCfg: None | DatasetConfig = None,
):
    randomCrop = RandomCrop(crop_size, p=1.0)
    minimum_roi_area = int(crop_size[0] * crop_size[1] * MINIMUM_ROI_AREA_RATE)

    if p4extraStaticObj != 0.0:
        staticObj_selector = StaticObjSelector(extraStaticObj_CV_SET, datasetCfg, scale_rate=(0.75, 1.5), max_objScale2Frame=0.35)

    def __PTZPanCrop(
        frames: torch.Tensor,
        empty_frames: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        video_id: int,
        gap: int = 5,
        roi_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        not_do_targets = random.random() > p4targets
        not_do_others = random.random() > p4others

        cate_id = get_VID2CID(video_id)
        if (
            cate_id == CAT2ID['PTZ']
            or (not_do_targets and not_do_others)
            or (random.random() > 0.5 and cate_id in [CAT2ID['badWeather'], CAT2ID['cameraJitter']])
        ):
            return randomCrop(frames, empty_frames, labels, features, roi_mask=roi_mask)

        if roi_mask is None:
            t, l, h, w = transforms.RandomCrop.get_params(frames, crop_size)
        else:
            t, l, h, w = get_tlhw_cropIncludeROI(minimum_roi_area, roi_mask, crop_size)

        if random.random() < p4extraStaticObj:
            num_staticObj = random.randint(1, 2)
            frames, labels, empty_frames, features = staticObj_selector.duplicate_apply(
                frames,
                empty_frames,
                labels,
                features,
                video_id,
                objCoverInfos=random.choices([True, False], k=num_staticObj),
                IntermittentObjInfos=[False] * num_staticObj,
            )

        # time line of PTZPanCrop
        #  do_others: |--empty------------|
        #  do_others:          |--recent--|
        # do_targets:                     |--curr--|

        # number of total update times
        update_time = (0 if not_do_others else empty_move_time) + (0 if not_do_targets else frames.shape[0])

        gap = max(5, gap // 4)  # normally, the PTZPan effect has less moving
        pixelMove_options = list(range(-gap, gap + 1))
        pixelMoveW_options = deepcopy(pixelMove_options)

        random.shuffle(pixelMoveW_options)
        for pixelMoveW in pixelMoveW_options:
            pixelMoveW_edge = l + pixelMoveW * update_time
            if pixelMoveW_edge >= 0 and frames.shape[-1] >= w + pixelMoveW_edge:
                break
        pixelMoveH_limit = len(pixelMove_options) // 4  # the vertical move is less on the CDNet2014
        pixelMoveH_options = pixelMove_options if pixelMoveH_limit == 0 else pixelMove_options[pixelMoveH_limit:-pixelMoveH_limit]
        random.shuffle(pixelMoveH_options)
        for pixelMoveH in pixelMoveH_options:
            pixelMoveH_edge = t + pixelMoveH * update_time
            if pixelMoveH_edge >= 0 and frames.shape[-2] >= h + pixelMoveH_edge:
                break

        abs_pixelMoveH, abs_pixelMoveW = abs(pixelMoveH), abs(pixelMoveW)
        if 0 == abs_pixelMoveH + abs_pixelMoveW:
            return randomCrop(frames, empty_frames, labels, features, roi_mask=roi_mask)

        # move_infos = [(t, l)...]
        move_infos = [
            (t + pixelMoveH * i, l + pixelMoveW * i)
            for i in (range(update_time) if random.random() > 0.5 else range(update_time - 1, -1, -1))
        ]

        # the moving effect for emptyBg & sequentialInfo
        if not_do_others:
            new_features = TF.F_t.crop(features, *move_infos[0], h, w)
            new_empty_frames = TF.F_t.crop(empty_frames, *move_infos[0], h, w)
        else:
            new_features = torch.zeros((*features.shape[:2], *crop_size), dtype=features.dtype)
            new_empty_frames = torch.zeros((*empty_frames.shape[:2], *crop_size), dtype=empty_frames.dtype)
            recent_move_start = empty_move_time - recent_move_time
            for t, l in move_infos[:recent_move_start]:
                new_empty_frames += TF.F_t.crop(empty_frames, t, l, h, w)
            for t, l in move_infos[recent_move_start:empty_move_time]:
                new_empty_frames += TF.F_t.crop(empty_frames, t, l, h, w)
                new_features[1] += TF.F_t.crop(features[1], t, l, h, w)

            new_empty_frames /= empty_move_time
            new_features[1] /= recent_move_time

            kernelHW = math.ceil(abs_pixelMoveH / 2) * 2 + 5, math.ceil(abs_pixelMoveW / 2) * 2 + 5
            new_empty_frames = TF.F_t.gaussian_blur(new_empty_frames, kernel_size=kernelHW, sigma=(0.1, 3.0))
            new_features[1] = TF.F_t.gaussian_blur(new_features[1], kernel_size=kernelHW, sigma=(0.1, 3.0))

        # the moving effect for detect_frames & labels
        if not_do_targets:
            new_frames = TF.F_t.crop(frames, *move_infos[-1], h, w)
            new_labels = TF.F_t.crop(labels, *move_infos[-1], h, w)
        else:
            new_frames = torch.zeros((*frames.shape[:2], *crop_size), dtype=frames.dtype)
            new_labels = torch.zeros((*labels.shape[:2], *crop_size), dtype=labels.dtype)
            for i, (t, l) in enumerate(move_infos[-frames.shape[0] :]):
                new_frames[i] = TF.F_t.crop(frames[i], t, l, h, w)
                new_labels[i] = TF.F_t.crop(labels[i], t, l, h, w)

        # * features: ERD -> empty, recent, recent - firstCurrent
        new_features[0] = new_empty_frames[0]
        new_features[2] = torch.abs(new_features[1] - new_frames[0])

        return new_frames, new_empty_frames, new_labels, new_features, roi_mask

    return __PTZPanCrop


def PTZZoomCrop(
    crop_size: Tuple[int] = (224, 224),
    min_orig_size: Tuple[int] = (112, 112),
    empty_move_time: int = 10,
    recent_move_time: int = 5,
    p4targets: float = 0.75,
    p4others: float = 0.9,
    p4extraStaticObj: float = 0.0,
    extraStaticObj_CV_SET: None | int = None,
    datasetCfg: None | DatasetConfig = None,
):
    randomCrop = RandomCrop(crop_size, p=1.0)
    minimum_roi_area = int(crop_size[0] * crop_size[1] * MINIMUM_ROI_AREA_RATE)

    if p4extraStaticObj != 0.0:
        staticObj_selector = StaticObjSelector(extraStaticObj_CV_SET, datasetCfg, scale_rate=(0.75, 1.5), max_objScale2Frame=0.35)

    def __PTZZoomCrop(
        frames: torch.Tensor,
        empty_frames: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        video_id: int,
        gap: int = 5,
        roi_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        not_do_targets = random.random() > p4targets
        not_do_others = random.random() > p4others

        cate_id = get_VID2CID(video_id)
        if (
            cate_id == CAT2ID['PTZ']
            or (not_do_targets and not_do_others)
            or (random.random() > 0.5 and cate_id in [CAT2ID['badWeather'], CAT2ID['cameraJitter']])
        ):
            return randomCrop(frames, empty_frames, labels, features, roi_mask=roi_mask)

        if roi_mask is None:
            t, l, h, w = transforms.RandomCrop.get_params(frames, crop_size)
        else:
            t, l, h, w = get_tlhw_cropIncludeROI(minimum_roi_area, roi_mask, crop_size)

        if random.random() < p4extraStaticObj:
            frames, labels, empty_frames, features = staticObj_selector.apply(frames, labels, empty_frames, features, video_id)

        # time line of PTZZoomCrop
        #  do_others: |--empty------------|
        #  do_others:          |--recent--|
        # do_targets:                     |--curr--|
        update_time = (0 if not_do_others else empty_move_time) + (0 if not_do_targets else frames.shape[0])

        pixelMove_options = list(range(1, max(3, gap // 4) + 1))
        random.shuffle(pixelMove_options)
        for pixelMove in pixelMove_options:
            max_pixelMove = pixelMove * update_time * 2 + max(min_orig_size)
            if frames.shape[-2] >= max_pixelMove and frames.shape[-1] >= max_pixelMove:
                break

        edge_t = t - max_pixelMove // 2
        edge_l = l - max_pixelMove // 2
        t = t - edge_t if edge_t < 0 else t
        l = l - edge_l if edge_l < 0 else l

        min_h = frames.shape[-2] - t - h - max_pixelMove
        min_w = frames.shape[-1] - l - w - max_pixelMove
        h = max(min_orig_size[0], h + min_h) if min_h < 0 else h
        w = max(min_orig_size[1], w + min_w) if min_w < 0 else w

        zoom_infos = [
            (t - pixelMove * i, l - pixelMove * i, h + pixelMove * i * 2, w + pixelMove * i * 2)
            for i in (range(update_time) if random.random() > 0.5 else range(update_time - 1, -1, -1))
        ]

        # the moving effect for emptyBg & sequentialInfo
        if not_do_others:
            new_features = TF.resized_crop(features, *zoom_infos[0], size=crop_size, antialias=True)
            new_empty_frames = TF.resized_crop(empty_frames, *zoom_infos[0], size=crop_size, antialias=True)
        else:
            new_features = torch.zeros((*features.shape[:2], *crop_size), dtype=features.dtype)
            new_empty_frames = torch.zeros((*empty_frames.shape[:2], *crop_size), dtype=empty_frames.dtype)
            recent_move_start = empty_move_time - recent_move_time
            for t, l, h, w in zoom_infos[:recent_move_start]:
                new_empty_frames += TF.resized_crop(empty_frames, t, l, h, w, size=crop_size, antialias=True)
            for t, l, h, w in zoom_infos[recent_move_start:empty_move_time]:
                new_empty_frames += TF.resized_crop(empty_frames, t, l, h, w, size=crop_size, antialias=True)
                new_features[1] += TF.resized_crop(features[1], t, l, h, w, size=crop_size, antialias=True)

            new_empty_frames /= empty_move_time
            new_features[1] /= recent_move_time

            kernelHW = [math.ceil(pixelMove / 2) * 2 + 5] * 2
            new_empty_frames = TF.F_t.gaussian_blur(new_empty_frames, kernel_size=kernelHW, sigma=(0.1, 3.0))
            new_features[1] = TF.F_t.gaussian_blur(new_features[1], kernel_size=kernelHW, sigma=(0.1, 3.0))

        # the moving effect for detect_frames & labels
        if not_do_targets:
            new_frames = TF.resized_crop(frames, *zoom_infos[-1], size=crop_size, antialias=True)
            new_labels = TF.resized_crop(labels, *zoom_infos[-1], size=crop_size, antialias=True)
        else:
            new_frames = torch.zeros((*frames.shape[:2], *crop_size), dtype=frames.dtype)
            new_labels = torch.zeros((*labels.shape[:2], *crop_size), dtype=labels.dtype)
            for i, (t, l, h, w) in enumerate(zoom_infos[-frames.shape[0] :]):
                new_frames[i] = TF.resized_crop(frames[i], t, l, h, w, crop_size, antialias=True)
                new_labels[i] = TF.resized_crop(labels[i], t, l, h, w, crop_size, interpolation=InterpolationMode.NEAREST)

        # * features: ERD -> empty, recent, recent - firstCurrent
        new_features[0] = new_empty_frames[0]
        new_features[2] = torch.abs(new_features[1] - new_frames[0])

        return new_frames, new_empty_frames, new_labels, new_features, roi_mask

    return __PTZZoomCrop


def SimilarStaticObjPaste(
    crop_size: Tuple[int] = (224, 224),
    coverMethod: str = 'mix',
    # max_zoom,
    duplicatedObj_range: Tuple[int] = (0, 5),
    extraStaticObj_CV_SET: None | int = None,
    datasetCfg: None | DatasetConfig = None,
):
    '''
    operator: str = `front` | `back` | `mix`
    '''

    coverMethod_map = {
        'front': lambda x: [True] * x,
        'back': lambda x: [False] * x,
        'mix': lambda x: random.choices([True, False], k=x),
    }

    cover_func: Callable[[int], int] = coverMethod_map[coverMethod]
    randomCrop = RandomCrop(crop_size, p=1.0)
    staticObj_selector = StaticObjSelector(extraStaticObj_CV_SET, datasetCfg, scale_rate=(0.75, 1.5), max_objScale2Frame=0.35)

    def __SimilarStaticObjPaste(
        frames: torch.Tensor,
        empty_frames: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        video_id: int,
        roi_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        cate_id = get_VID2CID(video_id)
        if cate_id != CAT2ID['PTZ'] or (random.random() > 0.5 and cate_id in [CAT2ID['badWeather'], CAT2ID['cameraJitter']]):
            num_staticObj = random.randint(*duplicatedObj_range)
            frames, labels, empty_frames, features = staticObj_selector.duplicate_apply(
                frames,
                empty_frames,
                labels,
                features,
                video_id,
                cover_func(num_staticObj),
                cover_func(num_staticObj) if num_staticObj < 3 else [False] * num_staticObj,
            )

        return randomCrop(frames, empty_frames, labels, features, roi_mask=roi_mask)

    return __SimilarStaticObjPaste


def AdditiveColorJitter(
    brightness: float | Tuple[float, float] = 0.0,
    contrast: float | Tuple[float, float] = 0.0,
    saturation: float | Tuple[float, float] = 0.0,
    hue: float | Tuple[float, float] = 0.0,
    p=0.9,
):
    ColorJitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __AdditiveColorJitter(
        frames: torch.Tensor,
        empty_frames: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        roi_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        # if torch.any(frames < 0):
        #     print("be, f < 0")
        # if frames.isnan().any():
        #     print("be, f nan")
        # if random.random() < p:
        #     # * features: ERD -> empty, recent, recent - firstCurrent
        #     empty_frames = ColorJitter(empty_frames)
        #     features[0] = empty_frames[0]

        #     concat_info = ColorJitter(torch.vstack((features[1:], frames)))
        #     features[1:], frames = concat_info[:2], concat_info[2:]

        # if frames.isnan().any():
        #     print("af, f nan")

        # TODO: this is a [Work Around] method
        fea = deepcopy(features)
        while True:
            if random.random() < p:
                # * features: ERD -> empty, recent, recent - firstCurrent
                e: torch.Tensor = ColorJitter(empty_frames)
                fea[0] = e[0]

                concat_info = ColorJitter(torch.vstack((fea[1:2], frames)))
                f: torch.Tensor
                fea[1], f = concat_info[0], concat_info[1:]

                if e.isnan().any() or fea.isnan().any() or f.isnan().any():
                    continue  # prevent any of input have `nan` exist

                fea[2] = torch.abs(fea[1] - f[0])

                empty_frames, features, frames = e, fea, f
            break

        return frames, empty_frames, labels, features, roi_mask

    return __AdditiveColorJitter


class GaussianNoise:
    def __init__(self, sigma: Tuple[float] | float = 0.01):
        self.__sigma = sigma
        self.sigma = self.select_sigma if isinstance(sigma, (tuple, list)) else sigma

    @property
    def select_sigma(self):
        return random.uniform(*self.__sigma)

    def __call__(
        self,
        frames: torch.Tensor,
        empty_frames: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        roi_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        frames += self.sigma * torch.randn_like(frames)

        return frames, empty_frames, labels, features, roi_mask
