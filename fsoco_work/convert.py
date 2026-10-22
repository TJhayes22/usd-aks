"""Convert the FSOCO bounding-box dataset (Supervisely format) to YOLO format.

Run from the activated .venv in fsoco_work:
    python convert.py

Tip: test on a small copy first (meta.json + a few images/annotations),
by pointing SRC at that copy.
"""
from pathlib import Path

import supervisely as sly

# --- edit these two paths if yours differ -----------------------------------
SRC = Path(r"C:\Users\Tyler\AKS_Root\fsoco_bounding_boxes_train")
DST = Path(r"C:\Users\Tyler\AKS_Root\fsoco_yolo")
# -----------------------------------------------------------------------------


def find_project_root(path: Path):
    """Return the folder that directly contains meta.json (SRC or one level down)."""
    if (path / "meta.json").exists():
        return path
    for child in sorted(path.iterdir()):
        if child.is_dir() and (child / "meta.json").exists():
            return child
    return None


def main():
    if not SRC.exists():
        raise SystemExit(f"Source folder not found: {SRC}")
    if DST.exists() and any(DST.iterdir()):
        raise SystemExit(f"Output folder is not empty: {DST}\nDelete it or pick a new DST.")

    root = find_project_root(SRC)
    if root is None:
        raise SystemExit(
            f"No meta.json found in {SRC} or its subfolders.\n"
            "Is the dataset extracted? Point SRC at the folder that contains meta.json."
        )

    print(f"Project root: {root}")
    print(f"Writing YOLO output to: {DST}")
    sly.convert.to_yolo(str(root), dest_dir=str(DST))

    labels = list(DST.rglob("*.txt"))
    images = [p for p in DST.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    print(f"Done. Found {len(images)} images and {len(labels)} .txt files in {DST}")
    print("Top-level contents of the output folder:")
    for item in sorted(DST.iterdir()):
        print("  ", item.name)


if __name__ == "__main__":
    main()