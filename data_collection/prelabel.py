"""
Pre-label frames for Label Studio so you only fix boxes instead of drawing them.

Runs YOLO-World (zero-shot, text prompt) or your own .pt model over a folder of images
and writes tasks.json with predictions for Label Studio to import.

Run it isolated, without touching your global Python:
  uv run --python 3.12 --with ultralytics prelabel.py --images to_label

Options:
  --model yolov8s-worldv2.pt           (default, zero-shot)  or  path/to/best.pt
  --prompts "rc car" "toy car"          text prompts (YOLO-World only)
  --keep-classes kart                   class names to keep (custom .pt only)
  --label rc_car                        Label Studio label to assign
  --conf 0.15                           low on purpose: deleting a wrong box is faster than drawing one
  --doc-root .                          must match LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--prompts", nargs="+", default=["rc car", "toy car", "remote control car"])
    ap.add_argument("--keep-classes", nargs="*", default=None)
    ap.add_argument("--label", default="rc_car")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--doc-root", default=".")
    ap.add_argument("--out", default="tasks.json")
    args = ap.parse_args()

    model = YOLO(args.model)
    is_world = "world" in Path(args.model).name.lower()
    if is_world:
        model.set_classes(args.prompts)

    doc_root = Path(args.doc_root).resolve()
    images = sorted(p for p in Path(args.images).resolve().iterdir() if p.suffix.lower() in IMG_EXT)
    tasks, n_boxes = [], 0

    for img in images:
        r = model.predict(str(img), conf=args.conf, verbose=False)[0]
        h, w = r.orig_shape
        results = []
        for box, cls, score in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if not is_world and args.keep_classes and r.names[int(cls)] not in args.keep_classes:
                continue
            x0, y0, x1, y1 = box
            results.append({
                "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                "original_width": w, "original_height": h, "image_rotation": 0,
                "score": float(score),
                "value": {
                    "x": 100 * x0 / w, "y": 100 * y0 / h,
                    "width": 100 * (x1 - x0) / w, "height": 100 * (y1 - y0) / h,
                    "rotation": 0, "rectanglelabels": [args.label],
                },
            })
        n_boxes += len(results)
        rel = img.relative_to(doc_root).as_posix()
        tasks.append({
            "data": {"image": f"/data/local-files/?d={rel}"},
            "predictions": [{"model_version": Path(args.model).stem, "result": results}],
        })

    Path(args.out).write_text(json.dumps(tasks, indent=1))
    print(f"{len(tasks)} images, {n_boxes} pre-labeled boxes -> {args.out}")


if __name__ == "__main__":
    main()