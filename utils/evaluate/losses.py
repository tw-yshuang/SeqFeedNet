"""Loss functions
"""

import torch
from torch import nn
from torch.nn import functional as F


class CDNet2014Convert(nn.Module):
    def __init__(self, nonvalid: int = -1, reduction: str = 'mean', *args, **kwargs) -> None:
        """
        Args:
        nonvalid (int): the unknown label value. convert by data_preprocess
        reduction (str, optional): Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        """
        super(CDNet2014Convert, self).__init__(*args, **kwargs)
        self.reduction = reduction
        self.nonvalid = nonvalid

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs, targets = inputs.flatten(1), targets.flatten(1)
        masks = torch.where(targets == self.nonvalid, 0, 1).type(torch.bool)

        match self.reduction:
            case 'rated':
                loss_ls = []
                for input, target, mask in zip(inputs, targets, masks):
                    loss_ls.append(self.loss_func(input, target, mask))
                    losses = torch.vstack(loss_ls).mean()
            case 'none':
                loss_ls = []
                for input, target, mask in zip(inputs, targets, masks):
                    loss_ls.append(self.loss_func(input, target, mask))
                    losses = torch.vstack(loss_ls)
            case 'sum':
                losses = self.loss_func(inputs, targets, masks).sum()
            case 'mean':
                losses = self.loss_func(inputs, targets, masks).mean()
            case _:
                raise ValueError(
                    f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
                )

        return losses

    def loss_func(self, inputs: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "There has two ways to implement it, \
                                  1: write a loss_func(self, ...) on the child class.\
                                  2: self.loss_func = self.xxx on the child class."
        )
        ...


class IOULoss4CDNet2014(CDNet2014Convert):
    def __init__(self, smooth: float = 100.0, nonvalid: int = -1, reduction: str = 'mean', *args, **kwargs) -> None:
        super(IOULoss4CDNet2014, self).__init__(nonvalid, reduction, *args, **kwargs)

        self.smooth = smooth
        self.loss_func = self.iou_loss_with_mask

    def iou_loss_with_mask(self, inputs: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor):
        """
        Args:
            inputs (Tensor): A float tensor of arbitrary shape.
                    The predictions for each example.
            targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class, 1 for the positive class, and -1 for unknown part).
            masks (Tensor): A bool tensor with the same shape as inputs. For extracting inputs and targets without unknown part.
        """
        input_val = torch.masked_select(inputs, masks)
        target_val = torch.masked_select(targets, masks)

        intersection = torch.sum(target_val * input_val)
        jac = (intersection + self.smooth) / (torch.sum(input_val) + torch.sum(target_val) - intersection + self.smooth)
        return (1 - jac) * self.smooth

    @staticmethod
    def __repr__():
        return 'IOULoss'


class FocalLoss4CDNet2014(CDNet2014Convert):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2,
        useSigmoid=False,
        nonvalid: int = -1,
        reduction: str = 'mean',
        *args,
        **kwargs,
    ) -> None:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            alpha (float): Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default: ``0.25``.
            gamma (float): Exponent of the modulating factor (1 - p_t) to
                    balance easy vs hard examples. Default: ``2``.
            useSigmoid (bool, optional):  use the sigmoid function as the first. In this paper, we plan to generate a mask that already has the sigmoid() at the end of the model. Default: ``False``
            reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                    ``'none'``: No reduction will be applied to the output.
                    ``'mean'``: The output will be averaged.
                    ``'sum'``: The output will be summed. Default: ``'none'``.
        """
        super(FocalLoss4CDNet2014, self).__init__(nonvalid, reduction, *args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.operator = nn.Sigmoid() if useSigmoid else nn.Identity()
        self.loss_func = self.focal_loss_with_mask

    def focal_loss_with_mask(self, inputs: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (Tensor): A float tensor of arbitrary shape.
                    The predictions for each example.
            targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class, 1 for the positive class, and -1 for unknown part).
            masks (Tensor): A bool tensor with the same shape as inputs. For extracting inputs and targets without unknown part.
        """
        input_val = torch.masked_select(inputs, masks)
        target_val = torch.masked_select(targets, masks)

        p: torch.Tensor = self.operator(input_val)

        ce_loss = F.binary_cross_entropy_with_logits(input_val, target_val, reduction="none")
        p_t = p * target_val + (1 - p) * (1 - target_val)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target_val + (1 - self.alpha) * (1 - target_val)
            loss = alpha_t * loss

        return loss if self.reduction == 'sum' else loss.mean()

    @staticmethod
    def __repr__():
        return 'FocalLoss'


class FocalLossRated4CDNet2014(CDNet2014Convert):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2,
        useSigmoid=False,
        nonvalid: int = -1,
        reduction: str = 'mean',
        *args,
        **kwargs,
    ) -> None:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            alpha (float): Weighting factor in range (0,1) to balance
                    positive vs negative examples or -1 for ignore. Default: ``0.25``.
            gamma (float): Exponent of the modulating factor (1 - p_t) to
                    balance easy vs hard examples. Default: ``2``.
            useSigmoid (bool, optional):  use the sigmoid function as the first. In this paper, we plan to generate a mask that already has the sigmoid() at the end of the model. Default: ``False``
            reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                    ``'none'``: No reduction will be applied to the output.
                    ``'mean'``: The output will be averaged.
                    ``'sum'``: The output will be summed. Default: ``'none'``.
        """
        super(FocalLossRated4CDNet2014, self).__init__(nonvalid, reduction, *args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.operator = nn.Sigmoid() if useSigmoid else nn.Identity()
        self.loss_func = self.focal_loss_with_mask_rated

    def focal_loss_with_mask_rated(self, inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (Tensor): A float tensor of arbitrary shape.
                    The predictions for each example.
            targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class, 1 for the positive class, and -1 for unknown part).
            mask (Tensor): A bool tensor with the same shape as inputs. For extracting inputs and targets without unknown part.
        """
        input_val = torch.masked_select(inputs, mask)
        target_val = torch.masked_select(targets, mask)

        p: torch.Tensor = self.operator(input_val)

        ce_loss = F.binary_cross_entropy_with_logits(input_val, target_val, reduction="none")
        p_t = p * target_val + (1 - p) * (1 - target_val)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target_val + (1 - self.alpha) * (1 - target_val)
            loss = alpha_t * loss

        return loss

    @staticmethod
    def __repr__():
        return 'FocalLossRated'


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

    result1 = IOULoss4CDNet2014(reduction='none').to(device)(gts, preds)
    result2 = IOULoss4CDNet2014(reduction='mean').to(device)(gts, preds)
    print(result1, result1.mean(), result2)
