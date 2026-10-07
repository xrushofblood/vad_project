"""
Exploratory analysis of continuous perception on assembly_2 (Phase 1).

Reads the per-frame patch labels saved by the scratch notebook (one 64x64 label grid per video frame) and the
event log annotated on the raw video (configs/assembly_2_events.json), and computes the measures reported in the
addendum of the multi-prototype study (V0 against V2). The definitions below are exactly those used there.

All parameters are provisional values chosen to look at the data, not justified thresholds:
  BODY_ZONE_DILATION = 3 patches  "on the body" = predicted body patches dilated by 3 patches
  PRESENCE_MIN = 3 patches        a piece is "present" in a frame (or zone) with at least 3 patches
  windows (frames)                see the constants below
They should be revisited (and justified) before any formal continuous-perception study.

Usage (from the project root):
  python continuous_analysis.py outputs/scratch/assembly_2_continuous_labels.npz \
                                outputs/scratch/assembly_2_continuous_labels_V2.npz
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

EVENTS_PATH = Path("configs/assembly_2_events.json")

BODY_ZONE_DILATION = 3        # patches: predicted body dilated by this many patches = "on the body" zone
PRESENCE_MIN = 3              # patches: minimum count for a piece to be "present"
BEFORE_PICK = (150, 5)        # window before a pick: frames [pick - 150, pick - 5)  (about 5 s, ending 5 frames before)
AFTER_MOUNT_DELAY = 30        # window after a mount: frames [mount + 30, END_OF_ASSEMBLY)
END_OF_ASSEMBLY = 2360        # last frame before the head turns to the monitor (event log notes)
BODY_ON_BASE = 345            # first frame after the body is mounted on the base (mount 336 + tolerance)
MOUTH_SLOT_WINDOW = (345, 540)  # body on base, mouth not yet picked (pick 543): the empty slot is visible
ARMS_MOUNTED_WINDOW = (2010, 2360)  # both arms mounted, hands away (arm_2 hands_off 2005)
FINAL_PAUSE = (2255, 2360)    # figure complete, no hands (event log notes)
END_WINDOW = (2400, 2444)     # monitor and controller in view (event log notes)
FPS = 30

PIECES = ["base", "hat", "eyes", "ear", "nose", "moustache", "mouth", "arm_1", "arm_2"]
CHECKED = ["hat", "eyes", "ear", "nose", "moustache", "mouth", "arm_1", "arm_2"]


def load_events(path=EVENTS_PATH):
    with open(path) as f:
        return json.load(f)["events"]


def analyse(labels_path, events):
    data = np.load(labels_path)
    labels = data["labels"]
    names = [str(n) for n in data["class_names"]]
    ids = {n: i for i, n in enumerate(names)}
    n_frames = len(labels)

    body = labels == ids["body"]
    zone = np.stack([ndimage.binary_dilation(b, iterations=BODY_ZONE_DILATION) if b.any() else np.zeros_like(b)
                     for b in body])
    on = {p: ((labels == ids[p]) & zone).sum(axis=(1, 2)) for p in PIECES}
    off = {p: ((labels == ids[p]) & ~zone).sum(axis=(1, 2)) for p in PIECES}

    first = {}
    for e in events:
        first.setdefault(e["piece"], e)          # first event of each piece (first ear for ear)
    last_mount = {e["piece"]: e["mount"] for e in events}   # last mount of each piece (second ear for ear)

    rows = {}
    for p in CHECKED:
        pick = first[p]["pick"]
        mount = last_mount[p]
        before = off[p][max(0, pick - BEFORE_PICK[0]):pick - BEFORE_PICK[1]]
        after = on[p][mount + AFTER_MOUNT_DELAY:END_OF_ASSEMBLY]
        before_on = on[p][BODY_ON_BASE:max(BODY_ON_BASE, pick - BEFORE_PICK[1])]
        rows[p] = {
            "seen on table before pick": (before >= PRESENCE_MIN).mean(),
            "seen on body after mount": (after >= PRESENCE_MIN).mean(),
            "on body before pick (false positive)": (before_on >= PRESENCE_MIN).mean(),
            "patches on body, final pause": on[p][FINAL_PAUSE[0]:FINAL_PAUSE[1]].mean(),
        }

    slot = on["mouth"][MOUTH_SLOT_WINDOW[0]:MOUTH_SLOT_WINDOW[1]]
    mouth_slot = {"frames with >= 1 mouth patch on body": (slot >= 1).mean(), "mean patches": slot.mean()}

    arms = {}
    a0, a1 = ARMS_MOUNTED_WINDOW
    for p in ["arm_1", "arm_2"]:
        mask = (labels[a0:a1] == ids[p]) & zone[a0:a1]
        left = right = 0
        for t in range(mask.shape[0]):
            body_cols = np.where(body[a0 + t])[1]
            if len(body_cols) == 0:
                continue
            centre = body_cols.mean()
            cols = np.where(mask[t])[1]
            left += int((cols < centre).sum())
            right += int((cols >= centre).sum())
        arms[p] = {"image left": left, "image right": right}

    moving = np.zeros(n_frames, dtype=bool)
    for e in events:
        moving[e["pick"]:e["hands_off"]] = True              # hands at work: from pick to hands off
    flicker = {}
    for p in PIECES:
        present = ((on[p] + off[p]) >= PRESENCE_MIN).astype(int)
        changes = np.abs(np.diff(present))
        rest, move = ~moving[1:], moving[1:]
        flicker[p] = {"rest (changes/s)": changes[rest].sum() / (rest.sum() / FPS),
                      "moving (changes/s)": changes[move].sum() / (move.sum() / FPS)}

    end = {p: off[p][END_WINDOW[0]:END_WINDOW[1]].mean() for p in PIECES}
    return {"per piece": pd.DataFrame(rows).T, "mouth slot": mouth_slot, "arms mounted": arms,
            "flicker": pd.DataFrame(flicker).T, "end (off-body patches)": end, "on": on, "off": off}


if __name__ == "__main__":
    events = load_events()
    results = {Path(p).stem: analyse(p, events) for p in sys.argv[1:]}
    pd.set_option("display.width", 200)
    for name, r in results.items():
        print(f"\n===== {name} =====")
        print(r["per piece"].round(2).to_string())
        print("mouth on empty slot:", {k: round(float(v), 2) for k, v in r["mouth slot"].items()})
        print("mounted arms (patches by side):", r["arms mounted"])
        print(r["flicker"].round(2).to_string())
        print("end of video, off-body patches:", {k: round(float(v), 1) for k, v in r["end (off-body patches)"].items()})