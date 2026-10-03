"""
Class definitions shared by all Phase 1 notebooks (single source of truth).

The order of CLASS_NAMES fixes the class indices and the color of each class in every plot.

The order is arbitrary, except that background must be first (index 0).
"""
import matplotlib.colors as mcolors

CLASS_NAMES = [
    "background", "body", "base", "hat", "eyes", "ear",
    "nose", "moustache", "mouth", "arm_1", "arm_2", "hand",
]
BACKGROUND_LABEL = "background"
HAND_LABEL = "hand"  # annotated so hand patches do not pollute other prototypes; never reported as a piece
PIECE_LABELS = frozenset(CLASS_NAMES) - {BACKGROUND_LABEL, HAND_LABEL}
LEGACY_LABELS = frozenset({"shoe", "pants"})  # old annotations, superseded by "base": always ignored

BACKGROUND_ID = CLASS_NAMES.index(BACKGROUND_LABEL)
IGNORE_ID = len(CLASS_NAMES)  # extra id for patches that are neither a piece nor background
PATCH_LABEL_NAMES = CLASS_NAMES + ["(ignored)"]
assert BACKGROUND_ID == 0, "patch_label_grid assumes that background has index 0"

# Hand-picked palette: 11 clearly distinct colors for the classes, light grey for background.
# Similar pieces (arm_1 / arm_2, nose / mouth / ear) get colors far apart on purpose.
CLASS_COLORS_HEX = {
    "background": "#d9d9d9",   # light grey
    "body":       "#4363d8",   # blue
    "base":       "#f58231",   # orange
    "hat":        "#911eb4",   # purple
    "eyes":       "#3cb44b",   # green
    "ear":        "#f032e6",   # magenta
    "nose":       "#e6194b",   # red
    "moustache":  "#9a6324",   # brown
    "mouth":      "#ffe119",   # yellow
    "arm_1":      "#42d4f4",   # cyan
    "arm_2":      "#808000",   # olive
    "hand":       "#469990",   # teal
}
assert set(CLASS_COLORS_HEX) == set(CLASS_NAMES), "Every class needs exactly one color"
CLASS_COLORS = {name: mcolors.to_rgb(hex_color) for name, hex_color in CLASS_COLORS_HEX.items()}
