"""Loss functions
"""

import torch

from torch.nn import functional as F


# todo: jaccard_loss use tp, fp, fn to build it


# code from https://github.com/ozantezcan/BSUV-Net-2.0/blob/69dac8b9a982a136bd1a59f4fb039983e6430c13/utils/losses.py#L31
def jaccard_loss(true, pred, smooth=100):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true (tensor): 1D ground truth tensor.
        preds (tensor): 1D prediction truth tensor.
        eps (int): Smoothing factor
    Returns:
        jacc_loss: the Jaccard loss.
    """
    intersection = torch.sum(true * pred)
    jac = (intersection + smooth) / (torch.sum(true) + torch.sum(pred) - intersection + smooth)
    return (1 - jac) * smooth


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
