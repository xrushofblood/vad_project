"""Label maps and overlays, used for test frames and video."""
import cv2
import numpy as np
from scipy import ndimage

from .classes import BACKGROUND_ID, CLASS_COLORS, CLASS_NAMES


def label_grid_to_rgb(label_grid):
    """Color each patch by its class; IGNORE_ID patches are black."""
    palette = np.array([CLASS_COLORS[name] for name in CLASS_NAMES] + [(0.0, 0.0, 0.0)])  # ignored = black
    return palette[label_grid]


def render_overlay(frame_rgb, labels, min_blob_patches):
    """Color predicted pieces over the frame and write the class name on blobs of at least
    min_blob_patches patches. Used for test frames and video."""
    size = (frame_rgb.shape[1], frame_rgb.shape[0])
    color = cv2.resize(label_grid_to_rgb(labels).astype(np.float32), size, interpolation=cv2.INTER_NEAREST)
    is_piece = cv2.resize((labels != BACKGROUND_ID).astype(np.uint8), size,
                          interpolation=cv2.INTER_NEAREST)[..., None].astype(bool)
    image = frame_rgb.astype(np.float32) / 255
    out = (np.where(is_piece, 0.3 * image + 0.7 * color, image) * 255).astype(np.uint8)

    scale_y, scale_x = size[1] / labels.shape[0], size[0] / labels.shape[1]
    for class_id, name in enumerate(CLASS_NAMES):
        if class_id == BACKGROUND_ID:
            continue
        blobs, n_blobs = ndimage.label(labels == class_id)
        for b in range(1, n_blobs + 1):
            rows, cols = np.where(blobs == b)
            if len(rows) < min_blob_patches:
                continue
            position = (int((cols.mean() + 0.5) * scale_x), int((rows.mean() + 0.5) * scale_y))
            cv2.putText(out, name, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out
