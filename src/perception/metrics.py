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



# --- Resolution study: arms merged, bootstrap, upsampling ------------------------------------

ARM_1_ID, ARM_2_ID = CLASS_NAMES.index("arm_1"), CLASS_NAMES.index("arm_2")
MERGED_NAMES = ["arm" if name == "arm_1" else name for name in CLASS_NAMES if name != "arm_2"]
MERGED_PIECES = (PIECE_LABELS - {"arm_1", "arm_2"}) | {"arm"}
_KEEP = [i for i in range(len(CLASS_NAMES)) if i != ARM_2_ID]
CI_PERCENTILES = (2.5, 97.5)   # 95% interval; written as numbers, so results are exactly reproducible


def upsample_labels(labels, factor):
    """Copy each label into a factor x factor block (nearest neighbour).
    E.g. a 32x32 grid at 512 -> the 64x64 grid of the 1024 ground truth (factor 2)."""
    return labels.repeat(factor, axis=0).repeat(factor, axis=1)


def merge_arms(confusion):
    """Confusion matrix with arm_1 and arm_2 counted as one class "arm" (in the place of arm_1).
    A patch of one arm predicted as the other counts as correct."""
    merged = confusion.copy()
    merged[ARM_1_ID, :] += merged[ARM_2_ID, :]
    merged[:, ARM_1_ID] += merged[:, ARM_2_ID]
    return merged[np.ix_(_KEEP, _KEEP)]


def merged_ids(class_ids):
    """Class indices translated to the arms-merged numbering (arm_2 -> arm, later classes shift by one)."""
    return {ARM_1_ID if i == ARM_2_ID else (i - 1 if i > ARM_2_ID else i) for i in class_ids}


def frame_iou(confusions, excluded_ids, frames=None, merge=False):
    """Per-class IoU over the given frames (all frames if None), counts summed before dividing.
    merge=True: arms merged (classes in MERGED_NAMES order)."""
    frames = range(len(confusions)) if frames is None else frames
    if merge:
        return iou_from_confusions([merge_arms(confusions[f]) for f in frames],
                                   [merged_ids(excluded_ids[f]) for f in frames])
    return iou_from_confusions([confusions[f] for f in frames], [excluded_ids[f] for f in frames])


def frame_miou(confusions, excluded_ids, frames=None, merge=False):
    """mIoU over the 10 pieces (or the 9 with arms merged) on the given frames."""
    iou = frame_iou(confusions, excluded_ids, frames, merge)
    return mean_iou(iou, MERGED_NAMES, MERGED_PIECES) if merge else mean_iou(iou)


def bootstrap_miou(confusions, excluded_ids, bootstrap_indices, merge=False):
    """mIoU on each bootstrap resample of the frames: array (n_resamples,).
    Use the same bootstrap_indices for two methods to compare them on paired resamples."""
    return np.array([frame_miou(confusions, excluded_ids, idx, merge) for idx in bootstrap_indices])


def bootstrap_class_iou(confusions, excluded_ids, bootstrap_indices, merge=False):
    """Per-class IoU on each bootstrap resample: array (n_resamples, n_classes).
    A class absent from a resample gets NaN."""
    return np.array([frame_iou(confusions, excluded_ids, idx, merge) for idx in bootstrap_indices])


def percentile_interval(values):
    """95% percentile interval along the first axis (NaN ignored), as (low, high).
    Applied to a paired difference (method A - method B on the same resamples), it is the
    paired bootstrap interval of the difference (Koehn 2004, Sect. 6)."""
    return tuple(np.nanpercentile(values, CI_PERCENTILES, axis=0))