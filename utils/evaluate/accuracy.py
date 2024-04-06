import torch


def calculate_acc_metrics(tp: int | torch.IntTensor, fp: int | torch.IntTensor, tn: int | torch.IntTensor, fn: int | torch.IntTensor):
    """
    The function calculates precision, recall, false negative rate, F1 score, and accuracy metrics based
    on true positives, false positives, true negatives, and false negatives.

    Args:
      tp (int | torch.IntTensor): True positives - the number of positive samples correctly classified
    as positive.
      fp (int | torch.IntTensor): The parameter "fp" stands for false positives, which refers to the
    number of instances that were incorrectly classified as positive when they are actually negative.
      tn (int | torch.IntTensor): True negatives (TN) are the number of observations that are correctly
    predicted as negative by a binary classification model. In other words, TN represents the number of
    true negative predictions.
      fn (int | torch.IntTensor): The parameter "fn" stands for "false negatives". It represents the
    number of instances that are actually positive but are incorrectly classified as negative.

    Returns:
      The function `calculate_acc_metrics` returns the following metrics: `precision`, `recall`, `false`
    negative rate (`fnr`), `f-score`, and `accuracy`.
    """
    prec = tp / (tp + fp + 0.0000001)
    recall = tp / (tp + fn + 0.0000001)
    fnr = 1 - recall

    f_score = 2 * (prec * recall) / (prec + recall + 0.000001)
    acc = (tp + tn) / (tp + fp + tn + fn)

    return prec, recall, fnr, f_score, acc
