"""
record_frames.py -- save frames from the OAK-D every N seconds, for labeling in Label Studio.

Put it in AKS_Root\\data_collection\\ (or its scripts\\ subfolder). Frames go to
AKS_Root\\data_collection\\raw\\<session>\\<session>_00001.jpg, one folder per recording
session (so you can split train/val by whole session later).

Examples:
    python record_frames.py --session 2026-10-20_lot_sunny
    python record_frames.py --session 2026-10-20_lot_sunny --interval 0.3 --preview
    python record_frames.py --session test --min-sharp 60      # skip blurry frames

Stop with Ctrl+C (or q in the preview window).
Preview keys: s = save a frame now, p = pause/resume, q = quit.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

_HERE = Path(__file__).resolve().parent
# works whether the script sits in data_collection\\ or data_collection\\scripts\\
DEFAULT_OUT = (_HERE.parent if _HERE.name == "scripts" else _HERE) / "raw"
FPS = 15                      # camera rate; saving is throttled by --interval anyway


def parse_size(s: str):
    w, h = s.lower().split("x")
    return int(w), int(h)


def small_gray(img):
    """Tiny grayscale copy used to detect near-duplicate frames."""
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.int16)


def sharpness(img):
    """Variance of the Laplacian: low = blurry. Computed on a 640-wide copy so it's size-independent."""
    h, w = img.shape[:2]
    g = cv2.cvtColor(cv2.resize(img, (640, int(640 * h / w))), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser(description="Record OAK-D frames for labeling")
    ap.add_argument("--session", default=datetime.now().strftime("%Y-%m-%d_%H%M"),
                    help="session folder name, e.g. 2026-10-20_lot_sunny")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="base output folder")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between saved frames")
    ap.add_argument("--size", type=parse_size, default=(1280, 720),
                    help="saved frame size WxH (16:9 matches the 512x288 deployment)")
    ap.add_argument("--min-sharp", type=float, default=0.0,
                    help="skip frames blurrier than this (try 40-80); 0 = off")
    ap.add_argument("--min-change", type=float, default=2.0,
                    help="skip frames too similar to the last saved one (mean pixel diff); 0 = off")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after this many saved frames; 0 = no limit")
    ap.add_argument("--preview", action="store_true", help="show a live preview window")
    args = ap.parse_args()

    session_dir = args.out / args.session
    session_dir.mkdir(parents=True, exist_ok=True)
    counter = len(list(session_dir.glob(f"{args.session}_*.jpg")))   # resume numbering if re-run
    (session_dir / "session_info.json").write_text(json.dumps({
        "session": args.session,
        "started": datetime.now().isoformat(timespec="seconds"),
        "interval_s": args.interval,
        "size": list(args.size),
        "min_sharp": args.min_sharp,
        "min_change": args.min_change,
    }, indent=2))

    saved = skipped_blur = skipped_dup = 0
    last_save = 0.0
    last_small = None
    paused = False
    show = args.preview

    print(f"Recording to {session_dir}  (every {args.interval}s, {args.size[0]}x{args.size[1]})")
    print("Ctrl+C to stop." + ("  Preview keys: s=save now, p=pause, q=quit" if show else ""))

    try:
        with dai.Pipeline() as pipeline:
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A, sensorFps=FPS)
            queue = cam.requestOutput(args.size, dai.ImgFrame.Type.BGR888p).createOutputQueue()
            pipeline.start()

            while pipeline.isRunning():
                img = queue.get().getCvFrame()
                key, force = 255, False

                if show:
                    try:
                        view = cv2.resize(img, (960, int(960 * img.shape[0] / img.shape[1])))
                        label = f"{args.session} | saved {saved}" + (" | PAUSED" if paused else "")
                        cv2.putText(view, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imshow("record_frames", view)
                        key = cv2.waitKey(1) & 0xFF
                    except cv2.error:
                        print("OpenCV window unavailable (headless build?) - continuing without preview")
                        show = False
                    if key == ord("q"):
                        break
                    if key == ord("p"):
                        paused = not paused
                    if key == ord("s"):
                        force = True

                if paused:
                    continue

                now = time.monotonic()
                if not force and (now - last_save) < args.interval:
                    continue
                last_save = now

                if not force:
                    if args.min_sharp > 0 and sharpness(img) < args.min_sharp:
                        skipped_blur += 1
                        continue
                    small = small_gray(img)
                    if (args.min_change > 0 and last_small is not None
                            and float(np.abs(small - last_small).mean()) < args.min_change):
                        skipped_dup += 1
                        continue
                else:
                    small = small_gray(img)

                counter += 1
                cv2.imwrite(str(session_dir / f"{args.session}_{counter:05d}.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                last_small = small
                saved += 1
                if saved % 20 == 0:
                    print(f"saved {saved} frames (skipped: {skipped_blur} blurry, {skipped_dup} duplicate)")
                if args.max_frames and saved >= args.max_frames:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if show:
            cv2.destroyAllWindows()

    print(f"\nDone. Saved {saved} frames to {session_dir}")
    print(f"Skipped: {skipped_blur} blurry, {skipped_dup} near-duplicate")


if __name__ == "__main__":
    main()