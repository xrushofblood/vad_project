"""
Contact sheets of the extracted frames: one image per video sequence.

Each sheet shows all the frames of one sequence as a grid of thumbnails, labelled with the
frame index and the second of the video (frames are extracted at 1 fps). Frames that already
have a labelme annotation (.json next to the .jpg) get a green border.

Usage (from the project root):
    python src/utils/make_contact_sheets.py
    python src/utils/make_contact_sheets.py --sequences assembly_1 caos --thumb 200 --cols 8
"""
import argparse
import glob
import os
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
FRAMES_DIR = os.path.join(PROJECT_ROOT, "data", "extracted_frames")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "contact_sheets")

LABEL_HEIGHT = 22        # pixels under each thumbnail for its label
TITLE_HEIGHT = 40        # pixels at the top of the sheet for the sequence name
BORDER = 4               # width of the green border marking annotated frames
ANNOTATED_COLOR = (40, 180, 70)


def load_font(size):
    """Arial if available (Windows), otherwise Pillow's default font."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def sequence_of(frame_name):
    """Frames are named <sequence>_f<index>: the sequence is everything before the last '_f'."""
    return frame_name.rsplit("_f", 1)[0]


def collect_frames(frames_dir):
    """Group frame paths by sequence, sorted by frame index."""
    groups = defaultdict(list)
    for path in glob.glob(os.path.join(frames_dir, "*.jpg")):
        name = os.path.splitext(os.path.basename(path))[0]
        if "_f" not in name:
            continue
        groups[sequence_of(name)].append(path)
    return {seq: sorted(paths) for seq, paths in sorted(groups.items())}


def make_sheet(sequence, paths, thumb, cols):
    """Build one contact sheet for one sequence."""
    rows = (len(paths) + cols - 1) // cols
    cell_h = thumb + LABEL_HEIGHT
    sheet = Image.new("RGB", (cols * thumb, TITLE_HEIGHT + rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    title_font, label_font = load_font(24), load_font(14)

    n_annotated = sum(os.path.exists(os.path.splitext(p)[0] + ".json") for p in paths)
    draw.text((10, 8), f"{sequence}  ({len(paths)} frames, {n_annotated} annotated)",
              fill="black", font=title_font)

    for i, path in enumerate(paths):
        x = (i % cols) * thumb
        y = TITLE_HEIGHT + (i // cols) * cell_h
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((thumb, thumb))
            sheet.paste(img, (x + (thumb - img.width) // 2, y + (thumb - img.height) // 2))

        if os.path.exists(os.path.splitext(path)[0] + ".json"):
            draw.rectangle([x, y, x + thumb - 1, y + thumb - 1], outline=ANNOTATED_COLOR, width=BORDER)

        index = os.path.splitext(os.path.basename(path))[0].rsplit("_f", 1)[1]
        draw.text((x + 6, y + thumb + 3), f"f{index} · {int(index)} s", fill="black", font=label_font)
    return sheet


def main():
    parser = argparse.ArgumentParser(description="One contact sheet per video sequence.")
    parser.add_argument("--sequences", nargs="*", help="only these sequences (default: all)")
    parser.add_argument("--thumb", type=int, default=160, help="thumbnail size in pixels")
    parser.add_argument("--cols", type=int, default=10, help="thumbnails per row")
    args = parser.parse_args()

    groups = collect_frames(FRAMES_DIR)
    if args.sequences:
        groups = {s: p for s, p in groups.items() if s in args.sequences}
    if not groups:
        print(f"No frames found in {FRAMES_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sequence, paths in groups.items():
        out_path = os.path.join(OUTPUT_DIR, f"{sequence}.jpg")
        make_sheet(sequence, paths, args.thumb, args.cols).save(out_path, quality=90)
        print(f"{sequence}: {len(paths)} frames -> {out_path}")


if __name__ == "__main__":
    main()