"""
Run lane detection on a source and show/print the results.

  python run_lanes.py --source oak
  python run_lanes.py --source recordings/track_run.mp4 --record out.mp4
  python run_lanes.py --source oak --headless          # on the Jetson, no display
"""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parent / "lane_config.json"   # beside this file, not the launch dir

try:
    from .lane_detector import LaneConfig, LaneDetector
    from .sources import open_source
except ImportError:  # run as a plain script
    from lane_detector import LaneConfig, LaneDetector
    from sources import open_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="oak")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", help="save the debug view to this .mp4")
    ap.add_argument("--print-every", type=int, default=10)
    args = ap.parse_args()

    det = LaneDetector(LaneConfig.load(args.config))
    writer = None
    times = []

    for i, frame in enumerate(open_source(args.source, args.width, args.height)):
        t0 = time.perf_counter()
        res = det.process(frame)
        times.append(time.perf_counter() - t0)

        if i % args.print_every == 0:
            ms = 1000 * np.mean(times[-50:])
            if res.ok:
                print(f"[{i:5d}] {ms:5.1f} ms  {res.source:10s} off {res.lateral_offset_m:+.3f} m  "
                      f"hdg {np.degrees(res.heading_error_rad):+5.1f} deg  "
                      f"k {res.curvature_1pm:+.2f} 1/m  target ({res.target_m[0]:+.2f}, {res.target_m[1]:.2f})  "
                      f"conf {res.confidence:.2f}")
            else:
                print(f"[{i:5d}] {ms:5.1f} ms  NO LANE")

        if args.headless and not args.record:
            continue
        overlay, bev = det.draw(res)
        bev = cv2.resize(bev, (int(bev.shape[1] * overlay.shape[0] / bev.shape[0]), overlay.shape[0]))
        view = np.hstack([overlay, bev])
        if args.record:
            if writer is None:
                writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 30, view.shape[1::-1])
            writer.write(view)
        if not args.headless:
            cv2.imshow("lanes", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("r"):
                det.reset()

    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
