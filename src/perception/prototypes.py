"""Prototype construction (normalized mean) and nearest-prototype classification."""
import json

import numpy as np

from .classes import BACKGROUND_ID, CLASS_NAMES


def l2_normalize(x, eps=1e-8):
    """Scale vectors to unit length along the last axis (only their direction is kept)."""
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def build_prototypes(reference_frames, class_names=CLASS_NAMES):
    """
    One prototype per class, computed as follows:
      1. every patch feature is L2-normalized (brought to length 1);
      2. all normalized patches of the class are pooled across ALL reference frames,
         each patch with the same weight (patch-weighted mean);
      3. their mean is L2-normalized again.
    This is the maximum-likelihood mean direction of a von Mises-Fisher distribution,
    the natural prototype when classification uses cosine similarity.
    Ignored patches (IGNORE_ID) never enter any prototype.

    Each reference frame is a dict with "features" (h, w, D) and "patch_labels" (h, w).
    Returns two dicts keyed by class name: prototypes (unit vectors) and patch counts.
    Classes without any patch are missing from both dicts.
    """
    vectors_per_class = {name: [] for name in class_names}
    for ref in reference_frames:
        features = l2_normalize(ref["features"])
        for class_id, name in enumerate(class_names):
            mask = ref["patch_labels"] == class_id
            if mask.any():
                vectors_per_class[name].append(features[mask])

    prototypes, patch_counts = {}, {}
    for name, chunks in vectors_per_class.items():
        if not chunks:
            continue
        pooled = np.concatenate(chunks)
        prototypes[name] = l2_normalize(pooled.mean(axis=0))
        patch_counts[name] = len(pooled)
    return prototypes, patch_counts


def stack_prototypes(prototypes, class_names=CLASS_NAMES):
    """(K, D) matrix of prototypes in class order. Every class must have a prototype."""
    missing = [name for name in class_names if name not in prototypes]
    assert not missing, f"No prototype for classes: {missing}"
    return np.stack([prototypes[name] for name in class_names])


def save_prototypes(path, prototypes, patch_counts, metadata, class_names=CLASS_NAMES):
    """Save prototypes (in class order), patch counts and metadata (a JSON-serializable dict)."""
    names = [name for name in class_names if name in prototypes]
    np.savez(
        path,
        prototypes=np.stack([prototypes[name] for name in names]),
        class_names=np.array(names),
        patch_counts=np.array([patch_counts[name] for name in names]),
        metadata=json.dumps(metadata),
    )


def load_prototypes(path):
    """Load a file written by save_prototypes. Returns class names, (K, D) matrix, patch counts, metadata."""
    data = np.load(path)
    return {
        "class_names": [str(name) for name in data["class_names"]],
        "prototypes": data["prototypes"],
        "patch_counts": data["patch_counts"],
        "metadata": json.loads(str(data["metadata"])),
    }


def assign_labels(similarities, margin, background_id=BACKGROUND_ID):
    """
    Labels from precomputed cosine similarities (h, w, K): nearest prototype, then background
    rejection. If the winner is not background, but its similarity exceeds the background
    similarity by less than `margin`, the patch is assigned to background.
    The margin only decides piece vs background, never between two pieces.
    Separate from classify_patches so that saved similarities can be relabelled without C-RADIO.
    """
    labels = similarities.argmax(axis=-1)
    best = similarities.max(axis=-1)
    rejected = (labels != background_id) & (best - similarities[..., background_id] < margin)
    return np.where(rejected, background_id, labels)


def classify_patches(features, prototype_matrix, margin, background_id=BACKGROUND_ID):
    """
    Classify every patch of one frame.

    1. Each patch feature is L2-normalized; its cosine similarity with each prototype is then
       a dot product, since prototypes are unit vectors.
    2. Each patch takes the most similar prototype (nearest class mean).
    3. Background rejection with `margin` (see assign_labels).

    Returns
    -------
    labels : (h, w) class indices into CLASS_NAMES
    similarities : (h, w, K) cosine similarity of each patch with each prototype
    """
    similarities = l2_normalize(features) @ prototype_matrix.T
    return assign_labels(similarities, margin, background_id), similarities
