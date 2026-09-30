"""Patch-level evaluation: confusion matrices, per-class IoU accumulated over frames, mean IoU."""
import numpy as np

from .classes import BACKGROUND_ID, CLASS_NAMES, IGNORE_ID, PIECE_LABELS
from scipy import ndimage


def confusion_matrix(gt_labels, pred_labels, n_classes=len(CLASS_NAMES)):
    """(K, K) patch counts of one frame: rows = ground truth, columns = prediction.
    Ignored ground-truth patches (IGNORE_ID) are skipped."""
    valid = gt_labels != IGNORE_ID
    codes = n_classes * gt_labels[valid] + pred_labels[valid]
    return np.bincount(codes, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def iou_from_confusions(confusions, excluded_ids):
    """Per-class IoU = TP / (TP + FP + FN), with TP, FP, FN summed over all frames.
    excluded_ids[f] lists the classes whose counts ignore frame f (e.g. base where it is out of definition).
    Classes absent from ground truth and predictions get NaN."""
    n_classes = confusions[0].shape[0]
    tp, fp, fn = (np.zeros(n_classes) for _ in range(3))
    for confusion, excluded in zip(confusions, excluded_ids):
        keep = ~np.isin(np.arange(n_classes), list(excluded))
        diagonal = np.diag(confusion)
        tp += keep * diagonal
        fp += keep * (confusion.sum(axis=0) - diagonal)
        fn += keep * (confusion.sum(axis=1) - diagonal)
    union = tp + fp + fn
    return np.where(union > 0, tp / np.maximum(union, 1), np.nan)


def mean_iou(iou, class_names=CLASS_NAMES, classes=PIECE_LABELS):
    """Mean IoU over the given classes (default: the 10 pieces), ignoring NaN."""
    return float(np.nanmean([iou[i] for i, name in enumerate(class_names) if name in classes]))

def ignore_boundary_ring(gt_labels, width=1):
    """Secondary ground truth: background patches within `width` patches of any annotated or ignored patch
    become ignored. Only background changes, so errors between pieces are counted exactly as before."""
    near_object = ndimage.binary_dilation(gt_labels != BACKGROUND_ID, structure=np.ones((3, 3)), iterations=width)
    tolerant = gt_labels.copy()
    tolerant[near_object & (gt_labels == BACKGROUND_ID)] = IGNORE_ID
    return tolerant