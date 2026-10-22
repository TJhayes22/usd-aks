"""Turn the converted FSOCO YOLO output into a single-class (cone) train/val dataset.

Run from the activated .venv in fsoco_work:
    python prepare_dataset.py

What it does:
  1. Finds every image and its YOLO .txt label inside SRC (layout-agnostic).
  2. Rewrites every class id to 0 (all FSOCO cone types -> one 'cone' class).
  3. Splits into train/val (by source folder when there are enough, else randomly).
  4. Optionally caps the image count and shrinks images (labels are normalized,
     so resizing doesn't change them) so the dataset is easy to upload to Colab.
  5. Writes dataset.yaml and a few preview images with boxes drawn, so you can
     eyeball that the labels are right.
"""
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]

# --- edit these ---------------------------------------------------------------
SRC = REPO / "fsoco_yolo"          # converter output
OUT = REPO / "cone_dataset"        # new dataset folder
MAX_IMAGES = 3000          # None = use everything
RESIZE_LONG_SIDE = 640     # None = keep original size
VAL_FRACTION = 0.15
SEED = 42
# -------------------------------------------------------------------------------

IMG_EXT = {".jpg", ".jpeg", ".png"}
SKIP_TXT = {"classes.txt", "obj.names", "labels.txt", "readme.txt"}


def find_label(img: Path, txt_by_stem: dict):
    """Locate the label file for an image."""
    candidates = [img.with_suffix(".txt")]
    parts = list(img.parts)
    for i, part in enumerate(parts):
        if part.lower() in {"images", "img", "image"}:
            for lab in ("labels", "label", "ann"):
                p = parts.copy()
                p[i] = lab
                candidates.append(Path(*p).with_suffix(".txt"))
    for c in candidates:
        if c.exists():
            return c
    hits = txt_by_stem.get(img.stem, [])
    return hits[0] if len(hits) == 1 else None


def read_boxes(label: Path):
    boxes = []
    for line in label.read_text().splitlines():
        tok = line.split()
        if len(tok) >= 5:
            try:
                cls = int(float(tok[0]))
                x, y, w, h = map(float, tok[1:5])
            except ValueError:
                continue
            boxes.append((cls, x, y, w, h))
    return boxes


def main():
    if not SRC.exists():
        raise SystemExit(f"Source not found: {SRC}")
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"Output folder not empty: {OUT}\nDelete it or pick a new OUT.")

    images = [p for p in SRC.rglob("*") if p.suffix.lower() in IMG_EXT]
    txt_by_stem = {}
    for t in SRC.rglob("*.txt"):
        if t.name.lower() not in SKIP_TXT:
            txt_by_stem.setdefault(t.stem, []).append(t)
    print(f"Found {len(images)} images and {sum(len(v) for v in txt_by_stem.values())} label files.")

    pairs, unmatched = [], 0
    class_counts = Counter()
    for img in images:
        lab = find_label(img, txt_by_stem)
        if lab is None:
            unmatched += 1
            continue
        boxes = read_boxes(lab)
        if not boxes:
            continue
        class_counts.update(b[0] for b in boxes)
        pairs.append((img, boxes))
    print(f"Matched {len(pairs)} images with at least one box ({unmatched} images had no label file).")
    print("Original class ids (before merging):", dict(sorted(class_counts.items())))
    if not pairs:
        raise SystemExit("Nothing matched. Send me the folder layout of SRC and I'll adjust the script.")

    rng = random.Random(SEED)
    rng.shuffle(pairs)
    if MAX_IMAGES and len(pairs) > MAX_IMAGES:
        pairs = pairs[:MAX_IMAGES]
        print(f"Capped to {MAX_IMAGES} images.")

    # split by top-level source folder (team/sequence) when there are enough groups
    groups = {}
    for img, boxes in pairs:
        key = img.relative_to(SRC).parts[0]
        groups.setdefault(key, []).append((img, boxes))
    if len(groups) >= 5:
        keys = sorted(groups)
        rng.shuffle(keys)
        n_val_groups = max(1, round(len(keys) * VAL_FRACTION))
        val_keys = set(keys[:n_val_groups])
        split = {"train": [], "val": []}
        for k, items in groups.items():
            split["val" if k in val_keys else "train"].extend(items)
        print(f"Split by source folder: {len(groups)} groups, {n_val_groups} held out for val.")
    else:
        n_val = max(1, int(len(pairs) * VAL_FRACTION))
        split = {"val": pairs[:n_val], "train": pairs[n_val:]}
        print("Few source folders, so split randomly by image.")

    for name, items in split.items():
        (OUT / "images" / name).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / name).mkdir(parents=True, exist_ok=True)
        for i, (img, boxes) in enumerate(items):
            stem = f"{name}_{i:05d}"
            dst_img = OUT / "images" / name / f"{stem}.jpg"
            with Image.open(img) as im:
                im = im.convert("RGB")
                if RESIZE_LONG_SIDE and max(im.size) > RESIZE_LONG_SIDE:
                    scale = RESIZE_LONG_SIDE / max(im.size)
                    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
                im.save(dst_img, quality=92)
            lines = [f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for _, x, y, w, h in boxes]
            (OUT / "labels" / name / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        print(f"  {name}: {len(items)} images")

    yaml_text = (
        f"path: {OUT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['cone']\n"
    )
    (OUT / "dataset.yaml").write_text(yaml_text)

    # previews with boxes drawn, to verify labels visually
    prev = OUT / "preview"
    prev.mkdir(exist_ok=True)
    for img_path in sorted((OUT / "images" / "train").glob("*.jpg"))[:8]:
        lab = OUT / "labels" / "train" / f"{img_path.stem}.txt"
        with Image.open(img_path) as im:
            d = ImageDraw.Draw(im)
            for line in lab.read_text().splitlines():
                _, x, y, w, h = map(float, line.split())
                W, H = im.size
                d.rectangle([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H],
                            outline=(255, 0, 0), width=2)
            im.save(prev / img_path.name)

    print(f"\nDone. Dataset at {OUT}")
    print(f"Open {prev} and check that the red boxes sit on cones.")


if __name__ == "__main__":
    main()