import random
from typing import List, Union, Callable

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


def monkeyPatch__call__(self, img: torch.Tensor):
    for t in self.transforms:
        if img.shape[1] == 1 and t.__class__.__name__ == transforms.Normalize.__name__:
            continue
        img = t(img)
    return img


transforms.Compose.__call__ = monkeyPatch__call__


class CustomCompose:
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, features: torch.Tensor, frames: torch.Tensor, labels: torch.Tensor | None = None):
        for t in self.transforms:
            if t.__module__ != transforms.transforms.__name__:
                frames, features, labels = t(frames, features, labels)
            else:
                features = t(features)
                frames = t(frames)

                if t.__class__.__name__ != transforms.Normalize.__name__:
                    labels = t(labels)

        return frames, features, labels

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class IterativeCustomCompose:
    def __init__(self, transforms: List[Callable], transform_img_size=(224, 224), device: str = 'cuda') -> None:
        '''
        transform_img_size: (H, W)
        '''
        self.compose = CustomCompose(transforms)
        self.transform_img_size = transform_img_size
        self.device = device

    def __call__(self, batch_features: torch.Tensor, batch_frames: torch.Tensor, batch_labels: torch.Tensor | None = None):
        process_batch_features = torch.zeros((*batch_features.shape[0:3], *self.transform_img_size), dtype=torch.float32).to(
            self.device
        )
        process_batch_frames = torch.zeros((*batch_frames.shape[0:3], *self.transform_img_size), dtype=torch.float32).to(self.device)
        process_batch_labels = torch.zeros((*batch_labels.shape[0:3], *self.transform_img_size), dtype=torch.float32).to(self.device)

        features: torch.Tensor
        frames: torch.Tensor
        labels: torch.Tensor
        for i, (features, frames, labels) in enumerate(zip(batch_features, batch_frames, batch_labels)):
            process_batch_features[i], process_batch_frames[i], process_batch_labels[i] = self.compose(features, frames, labels)

        return process_batch_features, process_batch_frames, process_batch_labels

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


def RandomCrop(crop_size=(224, 224), p=0.5):
    def __RandomCrop(frames: torch.Tensor, features: torch.Tensor, labels: torch.Tensor):
        if random.random() < p:
            t, l, h, w = transforms.RandomCrop.get_params(frames, crop_size)

            frames = TF.crop(frames, t, l, h, w)
            features = TF.crop(features, t, l, h, w)
            labels = TF.crop(features, t, l, h, w)

        return frames, features, labels

    return __RandomCrop


def RandomResizedCrop(
    sizeHW: List[int] = (224, 224),
    scale: List[float] = (0.6, 1.6),
    ratio: List[float] = (3.0 / 5.0, 2.0 / 1.0),
    p: float = 0.5,
):
    def __RandomResizedCrop(frames: torch.Tensor, features: torch.Tensor, labels: torch.Tensor):
        if random.random() < p:
            t, l, h, w = transforms.RandomResizedCrop.get_params(frames, scale, ratio)
            frames = TF.resized_crop(frames, t, l, h, w, size=sizeHW, antialias=True)
            features = TF.resized_crop(features, t, l, h, w, size=sizeHW, antialias=True)
            labels = TF.resized_crop(features, t, l, h, w, size=sizeHW, interpolation=InterpolationMode.NEAREST)
        else:
            w, h = frames.shape[-1], frames.shape[-2]
            frames = TF.resize(frames, size=sizeHW)
            features = TF.resize(features, size=sizeHW)
            # if isinstance(labels, torch.Tensor):
            labels = TF.resize(labels, size=sizeHW, interpolation=InterpolationMode.NEAREST)

        return frames, features, labels

    return __RandomResizedCrop
