"""
build_classification_dataset.py
Build a YOLOv8 image-classification dataset from FICS-PCB pre-cropped components.

FICS-PCB ships component crops sorted into class folders, e.g.:
    <board>/Microscope/components/front/1.5x/capacitors/*.png
    <board>/DSLR/components/<board>_front/ICs/*.png

This script gathers those crops into:
    <out>/train/<class>/*.png
    <out>/val/<class>/*.png
    <out>/test/<class>/*.png
assigning whole boards to a split (no board leaks across splits).

Usage:
    python build_classification_dataset.py --raw raw --out cls --boards s18 s28 s27 s29 s19
"""

import argparse
import shutil
from pathlib import Path

# Folder name (lowercased) -> canonical class. Three well-populated classes by default.
CLASS_MAP = {
    "capacitors": "capacitor", "capacitor": "capacitor",
    "resistors": "resistor", "resistor": "resistor",
    "ics": "ic", "ic": "ic",
    # The dataset also contains these, but they are too sparse for a clean split:
    # "inductors": "inductor", "transistors": "transistor", "diodes": "diode",
}

VAL_BOARDS = {"s29"}
TEST_BOARDS = {"s19"}
IMG_EXTS = {".png", ".jpg", ".jpeg"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="Folder containing unzipped board folders")
    p.add_argument("--out", required=True, help="Output classification dataset folder")
    p.add_argument("--boards", nargs="+", required=True, help="Board folder names to include")
    return p.parse_args()


def split_for_board(b):
    if b in VAL_BOARDS:
        return "val"
    if b in TEST_BOARDS:
        return "test"
    return "train"


def class_from_path(p):
    """Walk up from an image until a parent folder maps to a class (stop at 'components')."""
    for anc in p.parents:
        name = anc.name.lower()
        if name in CLASS_MAP:
            return CLASS_MAP[name]
        if name == "components":
            break
    return None


def main():
    args = parse_args()
    raw, out = Path(args.raw), Path(args.out)
    counts, copied = {}, 0

    for board in args.boards:
        bdir = raw / board
        if not bdir.is_dir():
            print("!! missing board:", board)
            continue
        split = split_for_board(board)
        for p in bdir.rglob("*"):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            if "__MACOSX" in str(p) or p.name.startswith("._"):
                continue
            if "components" not in [x.lower() for x in p.parts]:
                continue
            cls = class_from_path(p)
            if not cls:
                continue
            dst = out / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            unique = f"{board}__" + "_".join(p.relative_to(bdir).parts)
            try:
                shutil.copy(p, dst / unique)
                copied += 1
                counts.setdefault(split, {}).setdefault(cls, 0)
                counts[split][cls] += 1
            except Exception as e:
                print("  skip", p, e)

    print(f"\nCopied {copied} images\n")
    for s in ("train", "val", "test"):
        if s in counts:
            print(f"{s} ({sum(counts[s].values())} images):")
            for c, n in sorted(counts[s].items()):
                print(f"  {c:<12}: {n}")
            print()


if __name__ == "__main__":
    main()
