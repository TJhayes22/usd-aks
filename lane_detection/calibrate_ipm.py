"""
Bird's-eye (IPM) calibration.

1. Mount the camera where it will live on the car. Park on flat ground.
2. Lay a rectangle of known size flat in front of the car, roughly centered
   (tape outline or poster board). Measure:
     --width-m   rectangle width (left-right)
     --length-m  rectangle length (forward)
     --near-m    ground distance from directly below the camera to the near edge
3. Run, then click corners in order: NEAR-LEFT, NEAR-RIGHT, FAR-RIGHT, FAR-LEFT.
   Keys: r = reset clicks, s = save, q = quit.

Example:
  python calibrate_ipm.py --source oak --save-oak-intrinsics intrinsics.json \
      --width-m 0.6 --length-m 0.9 --near-m 0.3
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parent / "lane_config.json"   # beside this file, not the launch dir

try:
    from .lane_detector import LaneConfig, Undistorter, bev_size, compute_homography
    from .sources import open_source, save_oak_intrinsics
except ImportError:  # run as a plain script
    from lane_detector import LaneConfig, Undistorter, bev_size, compute_homography
    from sources import open_source, save_oak_intrinsics

LABELS = ["near-L", "near-R", "far-R", "far-L"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="oak")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width-m", type=float, required=True)
    ap.add_argument("--length-m", type=float, required=True)
    ap.add_argument("--near-m", type=float, required=True)
    ap.add_argument("--lateral-m", type=float, default=0.0, help="rect center, + = right of camera")
    ap.add_argument("--intrinsics", help="existing intrinsics JSON to undistort with")
    ap.add_argument("--save-oak-intrinsics", metavar="PATH", help="read K/D from the OAK and use them")
    args = ap.parse_args()

    cfg = LaneConfig.load(args.config)
    if args.save_oak_intrinsics:
        save_oak_intrinsics(args.save_oak_intrinsics, args.width, args.height)
        cfg.intrinsics_path = args.save_oak_intrinsics
    elif args.intrinsics:
        cfg.intrinsics_path = args.intrinsics
    undistort = Undistorter(cfg.intrinsics_path)

    cfg.rect_width_m, cfg.rect_length_m = args.width_m, args.length_m
    cfg.rect_near_dist_m, cfg.rect_lateral_offset_m = args.near_m, args.lateral_m

    clicks = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append([x, y])

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)

    frames = open_source(args.source, args.width, args.height, loop_image=True)
    frame = None
    for raw in frames:
        frame = undistort(raw)
        view = frame.copy()
        for i, p in enumerate(clicks):
            cv2.circle(view, tuple(p), 5, (0, 255, 255), -1)
            cv2.putText(view, LABELS[i], (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if len(clicks) > 1:
            cv2.polylines(view, [np.int32(clicks)], len(clicks) == 4, (0, 255, 255), 1)
        nxt = LABELS[len(clicks)] if len(clicks) < 4 else "done - press s to save"
        cv2.putText(view, f"click: {nxt}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("calibrate", view)

        if len(clicks) == 4:
            h, w = frame.shape[:2]
            cfg.src_pts, cfg.image_size = clicks, [w, h]
            M = compute_homography(cfg, (w, h))
            cv2.imshow("bird's-eye preview", cv2.warpPerspective(frame, M, bev_size(cfg)))

        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            clicks.clear()
        if k == ord("s") and len(clicks) == 4:
            cfg.save(args.config)
            print(f"Saved calibration to {args.config}")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
