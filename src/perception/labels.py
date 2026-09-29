"""From labelme polygons to patch labels (pixel map -> one label per 16x16 patch)."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from .classes import CLASS_NAMES, IGNORE_ID, LEGACY_LABELS


def sequence_of(frame_name):
    """Frames are named <sequence>_f<index>: the sequence is everything before the last '_f'."""
    return frame_name.rsplit("_f", 1)[0]


def load_annotation(json_path):
    """Read one labelme annotation file."""
    with open(json_path) as f:
        return json.load(f)


def check_annotation(image_path, json_path):
    """
    Check one annotated frame: both files exist, image and annotation have the same size,
    and every label is an official class or a legacy label.
    Returns the set of annotated labels.
    """
    image_path, json_path = Path(image_path), Path(json_path)
    assert image_path.exists(), f"Missing image: {image_path}"
    assert json_path.exists(), f"Missing annotation: {json_path}"

    annotation = load_annotation(json_path)
    with Image.open(image_path) as img:
        assert (img.width, img.height) == (annotation["imageWidth"], annotation["imageHeight"]), \
            f"Image/annotation size mismatch in {json_path.stem}"

    annotated = {shape["label"] for shape in annotation["shapes"]}
    unknown = annotated - set(CLASS_NAMES) - LEGACY_LABELS
    assert not unknown, f"Unknown labels in {json_path.stem}: {unknown}"
    return annotated


def rasterize_annotation(json_path, use_labels, target_size):
    """
    Turn the labelme polygons of one frame into a pixel map at model-input resolution.
    Each pixel holds a class index (0 = background, i.e. not annotated) or IGNORE_ID
    (polygons whose label is not in use_labels).
    Polygons are drawn from largest to smallest, so a piece mounted on top of another
    (e.g. eyes on the body) always wins, regardless of the order in the JSON file.
    """
    annotation = load_annotation(json_path)

    pixel_map = np.zeros((annotation["imageHeight"], annotation["imageWidth"]), dtype=np.uint8)
    shapes = sorted(annotation["shapes"], key=lambda s: Polygon(s["points"]).area, reverse=True)

    for shape in shapes:
        assert shape.get("shape_type", "polygon") == "polygon", f"Non-polygon shape in {Path(json_path).stem}"
        label = shape["label"]
        value = CLASS_NAMES.index(label) if label in use_labels else IGNORE_ID
        points = np.round(np.array(shape["points"])).astype(np.int32)
        cv2.fillPoly(pixel_map, [points], int(value))

    return cv2.resize(pixel_map, target_size, interpolation=cv2.INTER_NEAREST)


def patch_label_grid(pixel_map, patch_size, min_coverage):
    """
    Assign one label per patch from the pixel map.
    - piece label : the most frequent class is a piece and covers >= min_coverage of the patch
    - background  : the patch is 100% unannotated
    - ignored     : everything else (mixed boundary patches, ignored polygons)
    Returns the label grid (h_patches, w_patches) and the per-class pixel fractions.
    """
    H, W = pixel_map.shape
    h_p, w_p = H // patch_size, W // patch_size
    blocks = (pixel_map[:h_p * patch_size, :w_p * patch_size]
              .reshape(h_p, patch_size, w_p, patch_size)
              .transpose(0, 2, 1, 3)
              .reshape(h_p, w_p, -1))                                   # (h_p, w_p, patch_size**2 pixels)

    fractions = np.stack([(blocks == i).mean(axis=-1) for i in range(IGNORE_ID + 1)], axis=-1)
    majority = fractions.argmax(axis=-1)
    majority_fraction = fractions.max(axis=-1)

    labels = np.full((h_p, w_p), IGNORE_ID, dtype=np.int32)
    labels[fractions[..., 0] == 1.0] = 0                                # pure background
    is_piece = (majority != 0) & (majority != IGNORE_ID) & (majority_fraction >= min_coverage)
    labels[is_piece] = majority[is_piece]
    return labels, fractions
