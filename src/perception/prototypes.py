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
    This is the maximum-likelihood mean direction of a von Mises-Fisher distribution
    (Banerjee et al. 2005, Sect. 2.2).
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


# --- Multi-prototype study: several prototypes per class ---------------------------------------

def group_prototypes(reference_frames, class_names=CLASS_NAMES):
    """
    One prototype per (class, group): the normalized mean of the patches of that class and group, pooled over
    all reference frames exactly as in build_prototypes (same frame order, patch-weighted mean, L2 normalization).
    With a single group per class the result equals build_prototypes.

    Each reference frame is a dict with "features" (h, w, D), "patch_labels" (h, w) and "patch_groups" (h, w),
    an integer grid giving the group of each patch (e.g. 0 = loose, 1 = mounted).
    Returns the (P, D) prototype matrix, the class index and the group of each row, and the patch count of each row.
    """
    pooled = {}
    for ref in reference_frames:
        features = l2_normalize(ref["features"])
        for class_id in range(len(class_names)):
            in_class = ref["patch_labels"] == class_id
            for group in np.unique(ref["patch_groups"][in_class]):
                mask = in_class & (ref["patch_groups"] == group)
                pooled.setdefault((class_id, int(group)), []).append(features[mask])
    keys = sorted(pooled)
    vectors = [np.concatenate(pooled[key]) for key in keys]
    prototypes = np.stack([l2_normalize(v.mean(axis=0)) for v in vectors])
    return (prototypes, np.array([k[0] for k in keys]), np.array([k[1] for k in keys]),
            np.array([len(v) for v in vectors]))


def patch_prototypes(reference_frames, class_names=CLASS_NAMES):
    """
    Every labelled patch of the reference frames as a prototype (nearest-neighbour variant).
    Ignored patches (IGNORE_ID) are never used.
    Returns the (P, D) matrix of unit vectors and the class index of each row.
    """
    vectors, owners = [], []
    for ref in reference_frames:
        features = l2_normalize(ref["features"])
        for class_id in range(len(class_names)):
            mask = ref["patch_labels"] == class_id
            if mask.any():
                vectors.append(features[mask])
                owners.append(np.full(int(mask.sum()), class_id))
    return np.concatenate(vectors), np.concatenate(owners)


def class_similarities(features, prototypes, owners, n_classes=len(CLASS_NAMES), chunk=4096):
    """
    Similarity of each patch with each class: the highest cosine similarity among the prototypes of the class.
    features (h, w, D); prototypes (P, D) unit vectors; owners (P,) class index of each prototype.
    Prototypes are processed in chunks of `chunk` rows to bound memory; this does not change the result.
    With one prototype per class, in class order, the result equals the similarities of classify_patches.
    Labels then come from assign_labels, with the same background margin as the single-prototype method.
    """
    x = l2_normalize(features)
    out = np.full(features.shape[:-1] + (n_classes,), -np.inf, dtype=np.float32)
    for start in range(0, len(prototypes), chunk):
        similarities = x @ prototypes[start:start + chunk].T
        chunk_owners = owners[start:start + chunk]
        for class_id in np.unique(chunk_owners):
            np.maximum(out[..., class_id], similarities[..., chunk_owners == class_id].max(axis=-1),
                       out=out[..., class_id])
    assert np.isfinite(out).all(), "Every class needs at least one prototype"
    return out