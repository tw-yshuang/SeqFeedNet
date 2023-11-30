import random
from typing import Dict, List, Union, Callable

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


def monkeyPatch4Compose__call__(self, img: torch.Tensor):
    for t in self.transforms:
        if img.shape[1] == 1:
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

    def __call__(self, frames: torch.Tensor, labels: torch.Tensor | None = None, features: torch.Tensor | None = None):
        for t in self.transforms:
            if t.__module__ != transforms.transforms.__name__:
                frames, labels, features = t(frames, labels, features)
            else:
                if t.__class__.__name__ == transforms.RandomChoice.__name__:
                    t = t()
                    if t.__module__ != transforms.transforms.__name__:
                        frames, labels, features = t(frames, labels, features)
                        continue

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

        return frames, labels, features

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

        self.parameter_order = ['frames', 'labels', 'features']

    def __call__(
        self,
        b_frames: torch.Tensor,
        b_labels: torch.Tensor | None = None,
        b_features: torch.Tensor | None = None,
        useBuffer: bool = False,
    ):
        b_dict: Dict[str, torch.Tensor] = {}
        for b_items, name in zip([b_frames, b_labels, b_features], self.parameter_order):
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
        return process_b_dict['frames'], process_b_dict['labels'], process_b_dict['features']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


def RandomCrop(crop_size=(224, 224), p=0.5):
    def __RandomCrop(frames: torch.Tensor, labels: torch.Tensor, features: torch.Tensor):
        if random.random() < p:
            t, l, h, w = transforms.RandomCrop.get_params(frames, crop_size)

            features = TF.crop(features, t, l, h, w)
            frames = TF.crop(frames, t, l, h, w)
            labels = TF.crop(labels, t, l, h, w)

        return frames, labels, features

    return __RandomCrop


def RandomResizedCrop(
    sizeHW: List[int] = (224, 224),
    scale: List[float] = (0.6, 1.6),
    ratio: List[float] = (3.0 / 5.0, 2.0 / 1.0),
    p: float = 0.5,
):
    def __RandomResizedCrop(frames: torch.Tensor, labels: torch.Tensor, features: torch.Tensor):
        if random.random() < p:
            t, l, h, w = transforms.RandomResizedCrop.get_params(frames, scale, ratio)
            features = TF.resized_crop(features, t, l, h, w, size=sizeHW, antialias=True)
            frames = TF.resized_crop(frames, t, l, h, w, size=sizeHW, antialias=True)
            labels = TF.resized_crop(labels, t, l, h, w, size=sizeHW, interpolation=InterpolationMode.NEAREST)
        else:
            w, h = frames.shape[-1], frames.shape[-2]
            features = TF.resize(features, size=sizeHW, antialias=True)
            frames = TF.resize(frames, size=sizeHW, antialias=True)
            # if isinstance(labels, torch.Tensor):
            labels = TF.resize(labels, size=sizeHW, interpolation=InterpolationMode.NEAREST)

        return frames, labels, features

    return __RandomResizedCrop
