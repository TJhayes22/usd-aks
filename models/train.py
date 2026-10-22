"""
Fine-tune YOLO11n on the merged cone/person/kart set.

Usage (from any folder; paths resolve next to this file):
    python train.py coco     # start from COCO weights (recommended)
    python train.py cone     # start from your cone weights (ablation for PDR)
"""
import sys
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
START = {
    "coco": HERE / "yolo11n.pt",
    "cone": HERE / "cones_best.pt",
}


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "coco"
    model = YOLO(str(START[variant]))

    model.train(
        data=str(HERE / "merged" / "data.yaml"),
        imgsz=640,          # train at 640, benchmark smaller on the OAK-D later
        epochs=100,
        batch=16,
        patience=20,
        seed=0,
        workers=4,          # set to 0 if it hangs or throws a paging-file error
        device=0,
        name=f"cone_person_kart_from_{variant}",
    )

    # per-class mAP on val: check cone recall didn't regress
    metrics = model.val()
    for i, name in enumerate(model.names.values()):
        print(f"{name}: mAP50={metrics.box.ap50[i]:.3f}  mAP50-95={metrics.box.ap[i]:.3f}")


# Required on Windows: dataloader workers re-import this file when they spawn
if __name__ == "__main__":
    main()