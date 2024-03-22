"""
Helper functions to compute performance metrics
"""

import numpy as np
import torch
from PIL.Image import Image


def get_metrics(pred_img: torch.Tensor, gt_img: torch.Tensor) \
        -> tuple[float, tuple[float, float, float], tuple[float, float, float]]:
    """
    Compute some metrics given the received tensors and the batch_size
    tensor are of the form [B, C, H, W]
    - B: number of element in batch
    - C: number of chanel of image in tensor
    - H: Height of image in tensor
    - W: Weight of image in tensor

    :param pred_img: The first image as a tensor, represents the predictions
    :param gt_img: The second image as a tensor, represents the ground truths
    :return:
    - accuracy: fraction of correctly classified elements
    - precision_positive: true positive predictions over positive predictions
    - recall_positive: true positive predictions over positive ground truths
    - f1_positive: f1 positive score
    - precision_negative: true negative predictions over negative predictions
    - recall_negative: true negative predictions over negative ground truths
    - f1_negative: f1 negative score
    """
    size1, size2 = pred_img.size(), gt_img.size()

    if size1 != size2:
        raise Exception(f'Both tensors should have the same size, '
                        f'tensor 1 has size {size1}, while tensor 2 has size {size2}')

    tp, tn, fp, fn = __compute_confusion_matrix(pred_img, gt_img)

    accuracy = ((tp + tn) / (tp + tn + fp + fn))
    positive_metrics = __compute_precision_recall_f1(tp, fp, fn)
    negative_metrics = __compute_precision_recall_f1(tn, fn, fp)

    return accuracy, positive_metrics, negative_metrics


def get_performance_distribution(img_folds: list[list[Image]], gt_folds: list[list[Image]], compute_pred_gt) \
        -> list[list[float]]:
    """
    Function that performs a performance distribution of the model, by making prediction of k fold validation
    images.

    :param img_folds: The image folds list
    :param gt_folds: The groundtruth fold list
    :param compute_pred_gt: Function that takes an image fold and a groundtruth fold, and return a prediction tensor
    and a groundtruth tensor.

    :return: The performance metrics of each k_fold as a list
    """

    # Mean performances
    mean_accuracy = []
    mean_pos_precision = []
    mean_pos_recall = []
    mean_pos_f1 = []
    mean_neg_precision = []
    mean_neg_recall = []
    mean_neg_f1 = []

    for img_fold, gt_fold in zip(img_folds, gt_folds):

        pred_tensor, gt_tensor = compute_pred_gt(img_fold, gt_fold)

        accuracy, (p_p, r_p, f1_p), (p_n, r_n, f1_n) = get_metrics(pred_tensor, gt_tensor.round())

        # Update average metrics
        mean_accuracy.append(accuracy)
        mean_pos_precision.append(p_p)
        mean_pos_recall.append(r_p)
        mean_pos_f1.append(f1_p)
        mean_neg_precision.append(p_n)
        mean_neg_recall.append(r_n)
        mean_neg_f1.append(f1_n)

    return [mean_accuracy,
            mean_pos_precision, mean_pos_recall, mean_pos_f1,
            mean_neg_precision, mean_neg_recall, mean_neg_f1]


def __compute_confusion_matrix(pred_img: torch.Tensor, gt_img: torch.Tensor) -> tuple[int, int, int, int]:
    """
    Compute the confusion matrix for the given predictions and ground truths,
    with positive class having value 1 negative class having value 0

    :param pred_img: The first image as a tensor, represents the predictions
    :param gt_img: The second image as a tensor, represents the ground truths
    :return:
    - tp: number of positive predictions that have positive ground truths
    - tn: number of negative predictions that have negative ground truths
    - fp: number of positive predictions that have negative ground truths
    - fn: number of negative predictions that have positive ground truths
    """
    mask_same = pred_img == gt_img
    same = pred_img[mask_same]
    diff = pred_img[~mask_same]

    tp = (same == 1).sum().item()  # positive (road) classified as such
    tn = (same == 0).sum().item()  # negative (no road) classified as such
    fp = (diff == 1).sum().item()  # classified as positive (road), but was not a road
    fn = (diff == 0).sum().item()  # classified as negative (no road), but was road

    return tp, tn, fp, fn


def __compute_precision_recall_f1(t_pn, f_pn, f_np) -> tuple[float, float, float]:
    """
    Compute the positive/negative metrics, depending on the input

    :param t_pn: the number of true positives/negatives
    :param f_pn: the number of false positives/negatives
    :param f_np: the number of false negatives/positives
    :return:
    - precision_pn: the positive/negative precision
    - recall_pn: the positive/negative recall
    - f1_pn: the positive/negative f1 score
    """
    precision_pn = t_pn / (t_pn + f_pn) if (t_pn + f_pn) != 0 else np.nan
    recall_pn = t_pn / (t_pn + f_np) if (t_pn + f_np) != 0 else np.nan
    f1_pn = 2 * (precision_pn * recall_pn) / (precision_pn + recall_pn) \
        if (precision_pn + recall_pn) != 0 else np.nan

    return precision_pn, recall_pn, f1_pn
