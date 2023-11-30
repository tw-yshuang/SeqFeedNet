"""Loss functions
"""

import torch
from torch import nn
from torch.nn import functional as F


class CDNet2014_JaccardLoss(nn.Module):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']

    def __init__(self, smooth: float = 100.0, nonvalid=-1, reduction: str = 'mean', size_average=None, reduce=None):
        """
        Args:
        reduction (str, optional): Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied, ``'mean'``: the weighted mean of the output is taken, ``'sum'``: the output will be summed.
            Note: :attr:`size_average` and :attr:`reduce` are in the process of being deprecated, and in the meantime, specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        """
        super(CDNet2014_JaccardLoss, self).__init__()
        self.reduction = reduction
        self.smooth = smooth
        self.nonvalid = nonvalid

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs, targets = inputs.flatten(1), targets.flatten(1)
        masks = torch.where(inputs == self.nonvalid, 0, 1).type(torch.bool)

        losses: torch.Tensor
        match self.reduction:
            case 'none':
                losses = torch.zeros((inputs.shape[0], 1), dtype=torch.float32).to(inputs.device)
                for loss, target, input, mask in zip(losses, targets, inputs, masks):
                    loss[:] = self.jaccard_loss_with_mask(target, input, mask)
            case 'sum':
                losses = self.jaccard_loss_with_mask(targets, inputs, masks).sum()
            case 'mean':
                losses = self.jaccard_loss_with_mask(targets, inputs, masks).sum()
            case _:
                raise ValueError(
                    f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
                )

        return losses

    def jaccard_loss_with_mask(self, target: torch.Tensor, input: torch.Tensor, mask: torch.Tensor):
        input_val = torch.masked_select(target, mask)
        target_val = torch.masked_select(input, mask)

        intersection = torch.sum(target_val * input_val)
        jac = (intersection + self.smooth) / (torch.sum(target) + torch.sum(input) - intersection + self.smooth)
        return (1 - jac) * self.smooth

    def __repr__(self):
        return 'JaccardLoss'


# code from https://github.com/ozantezcan/BSUV-Net-2.0/blob/69dac8b9a982a136bd1a59f4fb039983e6430c13/utils/losses.py#L49
def weighted_crossentropy(true, pred, weight_pos=15, weight_neg=1):
    """Weighted cross entropy between ground truth and predictions
    Args:
        true (tensor): 1D ground truth tensor.
        preds (tensor): 1D prediction truth tensor.
    Returns:
        (tensor): Weighted CE.
    """
    bce = (true * pred.log()) + ((1 - true) * (1 - pred).log())  # Binary cross-entropy

    # Weighting for class imbalance
    weight_vector = true * weight_pos + (1.0 - true) * weight_neg
    weighted_bce = weight_vector * bce
    return -torch.mean(weighted_bce)


if __name__ == '__main__':
    import os, sys
    import cv2
    from pathlib import Path
    import torchvision.transforms as tvtf

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from utils.data_preprocess import CDNet2014Preprocess

    preprocess = CDNet2014Preprocess((224, 224))
    gts_pth = "/root/Work/fork-BGS/BSUV-Net-2.0/dataset/currentFr/baseline/highway/groundtruth"
    imgs_name = sorted(os.listdir(gts_pth))

    gts = list()
    vid_indices = list()
    for i, img_name in enumerate(imgs_name[699:]):
        if img_name.split('.')[-1] != 'png':
            continue
        elif i == 5:
            break

        img = cv2.imread(os.path.join(gts_pth, img_name), cv2.IMREAD_GRAYSCALE)
        gt = preprocess(img)
        gts.append(tvtf.ToTensor()(gt.copy()))
        vid_indices.append(11)

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    gts = torch.cat(gts, dim=0).reshape(len(gts), 1, 224, 224).to(device=device)
    preds = torch.zeros_like(gts, device=device)
    vid_indices = torch.tensor(vid_indices).reshape(5, 1).to(device=device, dtype=torch.int32)

    result1 = CDNet2014_JaccardLoss(reduction='none').to(device)(gts, preds)
    result2 = CDNet2014_JaccardLoss(reduction='mean').to(device)(gts, preds)
    print(result1, result1.mean(), result2)
