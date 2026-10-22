"""
Merge a Label Studio "YOLO with Images" export into our dataset.
Remaps chosen classes to our class IDs and splits train/val BY RECORDING SESSION
(so near-identical frames never land in both splits).

Frames must be named with a session tag, e.g. s03_frame_000123.jpg
(Label Studio may prepend a hash like 1a2b3c4d-s03_frame_000123.jpg; that's handled).

Example (Colab):
  python merge_labelstudio.py \
      --src /content/ls_export \
      --dst /content/drive/MyDrive/cone_dataset \
      --map rc_car=2 \
      --val-sessions s02 s05

  --map           ls_class=our_id, repeatable (check OUR data.yaml for the kart ID!)
  --val-sessions  sessions that go to val; everything else goes to train
  --no-empty      drop frames with no boxes (default keeps them as negatives)
"""
import argparse
import re
import shutil
from collections import Counter
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--map", nargs="+", required=True, metavar="NAME=ID")
    ap.add_argument("--val-sessions", nargs="*", default=[])
    ap.add_argument("--prefix", default="own")
    ap.add_argument("--no-empty", action="store_true")
    ap.add_argument("--session-regex", default=r"(s\d+)_",
                    help=r"regex whose group 1 is the session id, e.g. '(\d{4}-\d{2}-\d{2}_\d{4})_'")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    SESSION_RE = re.compile(args.session_regex)

    # Match the destination's layout: images/train (Ultralytics style) or train/images (Roboflow style)
    if (dst / "images").is_dir():
        if not any((dst / "images" / s).is_dir() for s in ("train", "val")):
            raise SystemExit(f"{dst / 'images'} has no train/ or val/ subfolders. Check data.yaml for the real layout.")
        out_dir = lambda split, kind: dst / kind / split
        print("Destination layout: images/<split>, labels/<split>")
    else:
        out_dir = lambda split, kind: dst / split / kind
        print("Destination layout: <split>/images, <split>/labels")
    ls_names = [n.strip() for n in (src / "classes.txt").read_text().splitlines() if n.strip()]
    wanted = dict(m.split("=") for m in args.map)
    missing = set(wanted) - set(ls_names)
    if missing:
        raise SystemExit(f"Not in classes.txt {ls_names}: {missing}")
    remap = {ls_names.index(n): int(i) for n, i in wanted.items()}
    print(f"Label Studio classes: {ls_names} -> remap {remap}")

    stats = Counter()
    for img in sorted((src / "images").iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        m = SESSION_RE.search(img.stem)
        if not m:
            print(f"  skip (no session tag): {img.name}")
            stats["no_session"] += 1
            continue
        session = m.group(1)
        split = "val" if session in args.val_sessions else "train"

        lbl = src / "labels" / f"{img.stem}.txt"
        lines = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5 and int(parts[0]) in remap:
                    lines.append(" ".join([str(remap[int(parts[0])]), *parts[1:]]))
        if not lines and args.no_empty:
            stats["dropped_empty"] += 1
            continue

        clean = img.stem[m.start():]  # strip Label Studio's hash prefix
        name = f"{args.prefix}_{clean}"
        out_dir(split, "images").mkdir(parents=True, exist_ok=True)
        out_dir(split, "labels").mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out_dir(split, "images") / f"{name}{img.suffix}")
        (out_dir(split, "labels") / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats[f"{split}_images"] += 1
        stats[f"{split}_boxes"] += len(lines)
        stats[f"{split}_negatives"] += not lines

    print(dict(stats))
    if not args.val_sessions:
        print("WARNING: no --val-sessions given, all frames went to train.")


if __name__ == "__main__":
    main()