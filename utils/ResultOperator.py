import os
from typing import Tuple

import numpy as np
import matplotlib.cm as cm
import torch
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

if __name__ == '__main__':
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from submodules.UsefulFileTools.FileOperator import check2create_dir


def tensor_to_jet(tensor: torch.Tensor):
    # Normalize the tensor
    norm_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    # Convert to numpy and apply colormap
    np_tensor = norm_tensor.cpu().numpy()
    colormap = cm.get_cmap('jet')
    colored_tensor: np.ndarray = colormap(np_tensor)[..., :3]

    # Convert to RGB and back to tensor
    # The last channel is alpha, which we discard for RGB image
    colored_tensor = torch.from_numpy(colored_tensor).permute(0, 3, 1, 2)

    return colored_tensor


class ResultOperator:
    def __init__(self, video_info: tuple, sizeHW: Tuple[int], taskDir: str = 'out') -> None:
        self.video_info = video_info
        self.sizeHW = sizeHW
        self.taskDir = taskDir
        # self.inverseNorm = transforms.Compose(
        #     [
        #         transforms.Normalize(mean=[0.0] * len(norm.std), std=[1 / value for value in norm.std]),
        #         transforms.Normalize(mean=[-value for value in norm.mean], std=[1.0] * len(norm.mean)),
        #     ]
        # )
        self.id = 1
        self.saveDir = f'{taskDir}/{self.video_info[0]}/{self.video_info[1]}'
        check2create_dir(f'{taskDir}/{self.video_info[0]}')
        check2create_dir(self.saveDir)

    def __call__(
        self,
        # frame: torch.Tensor,
        # label: torch.Tensor | None = None,
        pred_mask: torch.Tensor | None = None,
        pred: torch.Tensor | None = None,
        features: torch.Tensor | None = None,
    ):
        if self.id == 1:
            check2create_dir(f'{self.saveDir}/pred_mask') if pred_mask is not None else None
            check2create_dir(f'{self.saveDir}/pred') if pred is not None else None
            check2create_dir(f'{self.saveDir}/features') if features is not None else None

        if os.path.exists(f'{self.saveDir}/features/bin{self.id:06}.png'):
            self.id += 1
            return

        if pred_mask is not None:
            save_image(
                TF.F_t.resize(pred_mask.type(torch.float32), self.sizeHW, interpolation='nearest'),
                f'{self.saveDir}/pred_mask/bin{self.id:06}.png',
            )

        for i, (target, name) in enumerate(zip([pred, features], ['pred', 'features'])):
            if target is None:
                pass

            if i == 0:
                save_target = tensor_to_jet(target[0])
            else:
                save_target = torch.cat([tensor_to_jet(t) for t in target.permute(1, 0, 2, 3)])
            save_image(TF.F_t.resize(save_target, self.sizeHW, antialias=True), f'{self.saveDir}/{name}/bin{self.id:06}.png', nrow=9)

        self.id += 1
