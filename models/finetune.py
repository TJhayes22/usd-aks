"""
Fine-tune the current best cone/person/kart model on the merged set
(go-karts + RC cars now share class 2 "kart").

Usage:  yolo-env\Scripts\python finetune.py
"""
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
MERGED = HERE / "merged"
DATA = MERGED / "data.yaml"
OLD_WEIGHTS = HERE / "runs/detect/cone_person_kart_from_coco-3/weights/best.pt"
RUNS = HERE / "runs/detect"
RUN_NAME = "rc_finetune"


def check_gpu():
    if not torch.cuda.is_available():
        sys.exit(f"torch {torch.__version__} does not see a CUDA GPU; aborting.")
    print(f"GPU OK: {torch.cuda.get_device_name(0)} (torch {torch.__version__}, CUDA {torch.version.cuda})")


def check_data():
    cfg = yaml.safe_load(DATA.read_text())
    root = Path(cfg["path"]).resolve()
    if root != MERGED:
        sys.exit(f"data.yaml path: {root} does not point to {MERGED}")
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        if not (root / sub).is_dir():
            sys.exit(f"missing {root / sub}")
    print(f"data.yaml OK: path={root}, names={cfg['names']}")
    return cfg


def build_rc_val(cfg):
    """Write an image list + yaml for the RC-only val subset (own_* files)."""
    imgs = sorted((MERGED / "images/val").glob("own_*"))
    if not imgs:
        sys.exit("no own_* images in images/val")
    lst = MERGED / "rc_val.txt"
    lst.write_text("\n".join(str(p) for p in imgs) + "\n")
    rc_yaml = MERGED / "rc_val.yaml"
    rc_yaml.write_text(yaml.safe_dump({
        "path": str(MERGED), "train": str(lst), "val": str(lst), "names": cfg["names"],
    }, sort_keys=False))
    print(f"RC-only val set: {len(imgs)} images -> {rc_yaml}")
    return rc_yaml


def evaluate(weights, data, tag):
    m = YOLO(str(weights)).val(data=str(data), imgsz=640, batch=16, device=0,
                                plots=False, project=str(RUNS), name=f"{RUN_NAME}_eval_{tag}")
    kart = None
    idx = list(m.box.ap_class_index)
    if 2 in idx:
        i = idx.index(2)
        kart = (m.box.ap50[i], m.box.ap[i])
    return m.box.map50, m.box.map, kart


def main():
    check_gpu()
    cfg = check_data()
    rc_yaml = build_rc_val(cfg)

    model = YOLO(str(OLD_WEIGHTS))
    model.train(
        data=str(DATA),
        imgsz=640,
        epochs=40,
        batch=16,
        patience=15,
        seed=0,
        workers=4,          # set to 0 if it hangs or throws a paging-file error
        device=0,
        project=str(RUNS),
        name=RUN_NAME,
    )
    new_weights = Path(model.trainer.save_dir) / "weights/best.pt"

    rows = []
    for set_name, data, short in (("full val", DATA, "full"), ("RC-only val", rc_yaml, "rc")):
        for label, w in (("old", OLD_WEIGHTS), ("new", new_weights)):
            rows.append((set_name, label, *evaluate(w, data, f"{label}_{short}")))

    print("\n" + "=" * 72)
    print(f"{'set':<13}{'model':<7}{'mAP50':>8}{'mAP50-95':>10}{'kart mAP50':>12}{'kart 50-95':>12}")
    for set_name, label, map50, map5095, kart in rows:
        k50, k = (f"{kart[0]:.3f}", f"{kart[1]:.3f}") if kart else ("-", "-")
        print(f"{set_name:<13}{label:<7}{map50:>8.3f}{map5095:>10.3f}{k50:>12}{k:>12}")
    print("=" * 72)


# Required on Windows: dataloader workers re-import this file when they spawn
if __name__ == "__main__":
    main()
