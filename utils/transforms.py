import math, random
from typing import Dict, List, Callable, Tuple

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


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
    ):
        for t in self.transforms:
            if t.__module__ != transforms.transforms.__name__:
                # print(t)
                frames, empty_frames, labels, features, roi_mask = t(frames, empty_frames, labels, features, roi_mask=roi_mask)
            else:
                if t.__class__.__name__ == transforms.RandomChoice.__name__:
                    t = t()
                    if t.__module__ != transforms.transforms.__name__:
                        # print(t)
                        frames, empty_frames, labels, features, roi_mask = t(frames, empty_frames, labels, features, roi_mask=roi_mask)
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


def RandomCrop(crop_size=(224, 224), p=0.5):
    minimum_roi_area = int(crop_size[0] * crop_size[1] * MINIMUM_ROI_AREA_RATE)

    def __RandomCrop(*inputs, roi_mask: torch.Tensor | None = None):
        if random.random() < p:
            if roi_mask is None:
                t, l, h, w = transforms.RandomCrop.get_params(inputs[0], crop_size)
            else:
                t, l, h, w = get_tlhw_cropIncludeROI(minimum_roi_area, roi_mask, crop_size)

            inputs = [TF.F_t.crop(input, t, l, h, w) for input in inputs]

        return *inputs, roi_mask

    return __RandomCrop


def RandomHorizontalFlip(p=0.5):
    def __RandomHorizontalFlip(*inputs, roi_mask: torch.Tensor | None = None):
        if random.random() < p:
            inputs = [TF.F_t.hflip(input) for input in inputs]

        return *inputs, roi_mask

    return __RandomHorizontalFlip


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
    ):
        if random.random() < p:
            if roi_mask is None:
                t, l, h, w = transforms.RandomResizedCrop.get_params(frames, scale, ratio)
            else:
                t, l, h, w = get_tlhw_cropIncludeROI(minimum_roi_area, roi_mask, scale, ratio, CropMethod=transforms.RandomResizedCrop)

            features = TF.resized_crop(features, t, l, h, w, size=sizeHW, antialias=True)
            frames = TF.resized_crop(frames, t, l, h, w, size=sizeHW, antialias=True)
            empty_frames = TF.resized_crop(empty_frames, t, l, h, w, size=sizeHW, antialias=True)
            labels = TF.resized_crop(labels, t, l, h, w, size=sizeHW, interpolation=InterpolationMode.NEAREST)
        else:
            w, h = frames.shape[-1], frames.shape[-2]
            features = TF.F_t.resize(features, size=sizeHW, antialias=True)
            frames = TF.F_t.resize(frames, size=sizeHW, antialias=True)
            empty_frames = TF.F_t.resize(empty_frames, size=sizeHW, antialias=True)
            # if isinstance(labels, torch.Tensor):
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
    ):
        # new_frames = torch.zeros((*frames.shape[:2], *sizeHW), dtype=frames.dtype)
        # new_labels = torch.zeros((*labels.shape[:2], *sizeHW), dtype=labels.dtype)
        # new_empty_frames = torch.zeros((*empty_frames.shape[:2], *sizeHW), dtype=empty_frames.dtype)

        if random.random() > p:
            return randomCrop(frames, labels, empty_frames, features, roi_mask=roi_mask)

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
        new_features[2] = new_features[1] - frames[0]

        return frames, empty_frames, labels, new_features, roi_mask

    return __RandomShiftedCrop


def PTZPanCrop(
    crop_size: Tuple[int] = (224, 224),
    overlap_time: int = 5,
    max_pixelMoveH: int = 5,
    max_pixelMoveW: int = 5,
    p4targets: float = 0.75,
    p4others: float = 0.9,
):
    minimum_roi_area = int(crop_size[0] * crop_size[1] * MINIMUM_ROI_AREA_RATE)
    pixelMoveH_options = [*list(range(-max_pixelMoveH, 0)), *list(range(max_pixelMoveH + 1))]
    pixelMoveW_options = [*list(range(-max_pixelMoveW, 0)), *list(range(max_pixelMoveW + 1))]

    def __PTZPanCrop(
        frames: torch.Tensor,
        empty_frames: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        roi_mask: torch.Tensor | None = None,
    ):
        if roi_mask is None:
            t, l, h, w = transforms.RandomCrop.get_params(frames, crop_size)
        else:
            t, l, h, w = get_tlhw_cropIncludeROI(minimum_roi_area, roi_mask, crop_size)

        update_time = overlap_time - 1
        while True:
            pixelMoveH = random.choice(pixelMoveH_options)
            pixelMoveH_edge = pixelMoveH * update_time
            if t + pixelMoveH_edge > 0 and frames.shape[-2] > t + h + pixelMoveH_edge:
                break
        while True:
            pixelMoveW = random.choice(pixelMoveW_options)
            pixelMoveW_edge = pixelMoveW * update_time
            if l + pixelMoveW_edge > 0 and frames.shape[-1] > l + w + pixelMoveW_edge:
                break

        if random.random() > p4targets:
            new_frames = TF.F_t.crop(frames, t, l, h, w)
            new_labels = TF.F_t.crop(labels, t, l, h, w)
        else:
            new_frames = torch.zeros((*frames.shape[:2], *crop_size), dtype=frames.dtype)
            new_labels = torch.zeros((*labels.shape[:2], *crop_size), dtype=labels.dtype)
            for i in range(frames.shape[0]):
                new_frames[i] = TF.F_t.crop(frames[i], t + pixelMoveH * i * 2, l + pixelMoveW * i * 2, h, w)
                new_labels[i] = TF.F_t.crop(labels[i], t + pixelMoveH * i * 2, l + pixelMoveW * i * 2, h, w)

        if random.random() > p4others:
            new_features = TF.F_t.crop(features, t, l, h, w)
            new_empty_frames = TF.F_t.crop(empty_frames, t, l, h, w)
        else:
            new_features = torch.zeros((*features.shape[:2], *crop_size), dtype=features.dtype)
            new_empty_frames = torch.zeros((*empty_frames.shape[:2], *crop_size), dtype=empty_frames.dtype)
            new_features[1] = TF.F_t.crop(features[1], t, l, h, w)
            new_empty_frames[:] = TF.F_t.crop(empty_frames, t, l, h, w)
            for i in range(1, overlap_time):
                new_features[1] += TF.F_t.crop(features[1], t + pixelMoveH * i, l + pixelMoveW * i, h, w)
                new_empty_frames += TF.F_t.crop(empty_frames, t + pixelMoveH * i, l + pixelMoveW * i, h, w)

            new_empty_frames /= overlap_time
            new_features[1] /= overlap_time

            kernelHW = math.ceil(abs(pixelMoveH) / 2) * 2 + 1, math.ceil(abs(pixelMoveW) / 2) * 2 + 1
            new_empty_frames = TF.F_t.gaussian_blur(new_empty_frames, kernel_size=kernelHW, sigma=(0.1, 3.0))
            new_features[1] = TF.F_t.gaussian_blur(new_features[1], kernel_size=kernelHW, sigma=(0.1, 3.0))

        # * features: ERD -> empty, recent, recent - firstCurrent
        new_features[0] = new_empty_frames[0]
        new_features[2] = new_features[1] - new_frames[0]

        return new_frames, new_empty_frames, new_labels, new_features, roi_mask

    return __PTZPanCrop


def PTZZoomCrop(
    crop_size: Tuple[int] = (224, 224),
    overlap_time: int = 5,
    max_pixelMove: int = 5,
    p4targets: float = 0.75,
    p4others: float = 0.9,
):
    minimum_roi_area = int(crop_size[0] * crop_size[1] * MINIMUM_ROI_AREA_RATE)
    pixelMove_options = list(range(1, max_pixelMove + 1))
    update_time = overlap_time - 1

    def __PTZZoomCrop(
        frames: torch.Tensor,
        empty_frames: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        roi_mask: torch.Tensor | None = None,
    ):
        if roi_mask is None:
            t, l, h, w = transforms.RandomCrop.get_params(frames, crop_size)
        else:
            t, l, h, w = get_tlhw_cropIncludeROI(minimum_roi_area, roi_mask, crop_size)

        pixelMove = random.choice(pixelMove_options)

        if random.random() > p4targets:
            new_frames = TF.F_t.crop(frames, t, l, h, w)
            new_labels = TF.F_t.crop(labels, t, l, h, w)
        else:
            frameMove_edge = pixelMove * frames.shape[0]
            edge_t, edge_h = t - frameMove_edge * 2, frames.shape[-2] - t - h - frameMove_edge * 2
            edge_l, edge_w = l - frameMove_edge * 2, frames.shape[-1] - l - w - frameMove_edge * 2
            fix_edge = min(edge_t, edge_h, edge_l, edge_w)
            if fix_edge < 0:  # fix_edge to avoid black edge
                frame_t, frame_l, frame_h, frame_w = t - fix_edge, l - fix_edge, h + fix_edge * 2, w + fix_edge * 2
            else:
                frame_t, frame_l, frame_h, frame_w = t, l, h, w

            new_frames = torch.zeros((*frames.shape[:2], *crop_size), dtype=frames.dtype)
            new_labels = torch.zeros((*labels.shape[:2], *crop_size), dtype=labels.dtype)
            for i, frame_idx in enumerate(range(frames.shape[0]) if random.random() > 0.5 else range(frames.shape[0] - 1, -1, -1)):
                frameMove = pixelMove * i * 2
                new_frames[frame_idx] = TF.resized_crop(
                    frames[frame_idx],
                    frame_t - frameMove,
                    frame_l - frameMove,
                    frame_h + frameMove * 2,
                    frame_w + frameMove * 2,
                    size=crop_size,
                    antialias=True,
                )
                new_labels[frame_idx] = TF.resized_crop(
                    labels[frame_idx],
                    frame_t - frameMove,
                    frame_l - frameMove,
                    frame_h + frameMove * 2,
                    frame_w + frameMove * 2,
                    size=crop_size,
                    interpolation=InterpolationMode.NEAREST,
                )

        if random.random() > p4others:
            new_features = TF.F_t.crop(features, t, l, h, w)
            new_empty_frames = TF.F_t.crop(empty_frames, t, l, h, w)
        else:
            frameMove_edge = pixelMove * update_time
            edge_t, edge_h = t - frameMove_edge, frames.shape[-2] - t - h - frameMove_edge
            edge_l, edge_w = l - frameMove_edge, frames.shape[-1] - l - w - frameMove_edge
            fix_edge = min(edge_t, edge_h, edge_l, edge_w)
            if fix_edge < 0:  # fix_edge to avoid black edge
                t, l, h, w = t - fix_edge, l - fix_edge, h + fix_edge * 2, w + fix_edge * 2

            new_features = torch.zeros((*features.shape[:2], *crop_size), dtype=features.dtype)
            new_empty_frames = torch.zeros((*empty_frames.shape[:2], *crop_size), dtype=empty_frames.dtype)
            for i in range(overlap_time):
                tempMove = pixelMove * i
                new_features[1] += TF.resized_crop(
                    features[1],
                    t - tempMove,
                    l - tempMove,
                    h + tempMove * 2,
                    w + tempMove * 2,
                    size=crop_size,
                    antialias=True,
                )
                new_empty_frames += TF.resized_crop(
                    empty_frames,
                    t - tempMove,
                    l - tempMove,
                    h + tempMove * 2,
                    w + tempMove * 2,
                    size=crop_size,
                    antialias=True,
                )

            new_empty_frames /= overlap_time
            new_features[1] /= overlap_time

            kernelHW = math.ceil(abs(pixelMove) / 2) * 2 + 1, math.ceil(abs(pixelMove) / 2) * 2 + 1
            new_empty_frames = TF.F_t.gaussian_blur(new_empty_frames, kernel_size=kernelHW, sigma=(0.1, 3.0))
            new_features[1] = TF.F_t.gaussian_blur(new_features[1], kernel_size=kernelHW, sigma=(0.1, 3.0))

        # * features: ERD -> empty, recent, recent - firstCurrent
        new_features[0] = new_empty_frames[0]
        new_features[2] = new_features[1] - new_frames[0]

        return new_frames, new_empty_frames, new_labels, new_features, roi_mask

    return __PTZZoomCrop


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
    ):
        if random.random() < p:
            # * features: ERD -> empty, recent, recent - firstCurrent
            empty_frames = ColorJitter(empty_frames)
            features[0] = empty_frames[0]

            concat_info = ColorJitter(torch.vstack((features[1:], frames)))
            features[1:], frames = concat_info[:2], concat_info[2:]

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
    ):
        frames += self.sigma * torch.randn_like(frames)

        return frames, empty_frames, labels, features, roi_mask
