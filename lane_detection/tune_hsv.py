"""
Live HSV threshold tuner. Goal: tape lines solid white in the mask, everything else black.
Keys: space = pause/resume, s = save to config, q = quit.

  python tune_hsv.py --source oak
  python tune_hsv.py --source recordings/track_run.mp4
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parent / "lane_config.json"   # beside this file, not the launch dir

try:
    from .lane_detector import LaneConfig, Undistorter, threshold_hsv
    from .sources import open_source
except ImportError:  # run as a plain script
    from lane_detector import LaneConfig, Undistorter, threshold_hsv
    from sources import open_source

BARS = [("H lo", 0, 179), ("S lo", 1, 255), ("V lo", 2, 255), ("H hi", 0, 179), ("S hi", 1, 255), ("V hi", 2, 255)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="oak")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    cfg = LaneConfig.load(args.config)
    undistort = Undistorter(cfg.intrinsics_path)

    cv2.namedWindow("tune")
    init = list(cfg.hsv_lower) + list(cfg.hsv_upper)
    for (name, _, mx), v in zip(BARS, init):
        cv2.createTrackbar(name, "tune", int(v), mx, lambda _: None)

    frames = open_source(args.source, args.width, args.height, loop_image=True)
    frame, paused = None, False
    while True:
        if frame is None or not paused:
            try:
                frame = undistort(next(frames))
            except StopIteration:
                frames = open_source(args.source, args.width, args.height, loop_image=True)  # loop video
                continue

        vals = [cv2.getTrackbarPos(n, "tune") for n, _, _ in BARS]
        lo, hi = vals[:3], vals[3:]
        mask = threshold_hsv(frame, lo, hi)
        view = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        note = "hue wraps (red mode)" if lo[0] > hi[0] else ""
        cv2.putText(view, f"lo {lo} hi {hi} {note}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("tune", view)

        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            paused = not paused
        if k == ord("s"):
            cfg.hsv_lower, cfg.hsv_upper = lo, hi
            cfg.save(args.config)
            print(f"Saved hsv_lower={lo} hsv_upper={hi} to {args.config}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
