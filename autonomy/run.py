"""Run the stack with lane detection as the path source, from a recorded clip or the OAK-D.

    python -m autonomy.run --source clip.mp4                          # offline, simulated time
    python -m autonomy.run --source clip.mp4 --obstacles obstacles.csv --plot run.png
    python -m autonomy.run --source oak                               # live camera, real time (Ctrl+C stops)

RC and vehicle are mocks (armed in MANUAL, switched to AUTO when the run starts), so this
checks perception -> decisions without driving anything. --source oak opens the camera for
lanes only; it can't share the OAK-D with oak_obstacles.py.
"""
import argparse
import logging
import time
from collections import Counter
from pathlib import Path

from .adapters.csv_replay import CsvReplayPerception
from .adapters.lane_perception import LanePerception
from .adapters.mock_rc import MockRc
from .adapters.mock_vehicle import MockVehicle
from .config import load_config
from .loop import Loop, SimClock
from .paths import DATA, REPO
from .types import State

DEFAULT_LANE_CONFIG = REPO / "lane_detection" / "lane_config.json"
LEAD_IN_S = 0.5   # simulated time before the first frame, so the car can arm in MANUAL


def build(args, cfg):
    obstacles = CsvReplayPerception(args.obstacles, cfg) if args.obstacles else None
    perception = LanePerception.from_source(cfg, args.source, args.lane_config,
                                            args.width, args.height, obstacle_source=obstacles)
    live = perception.fps is None
    if live:
        t0 = time.monotonic()
        rc = MockRc([(t0, {"arm": True, "mode_auto": False}), (t0 + LEAD_IN_S, {"mode_auto": True})])
        loop = Loop(cfg, perception, rc, MockVehicle(cfg))
    else:
        clock = SimClock(-LEAD_IN_S)
        rc = MockRc([(-LEAD_IN_S, {"arm": True, "mode_auto": False}), (0.0, {"mode_auto": True})])
        loop = Loop(cfg, perception, rc, MockVehicle(cfg), clock=clock, sleep=clock.sleep)
    return perception, loop


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True, help="'oak', a webcam index, or a video/image file")
    ap.add_argument("--lane-config", type=Path, default=DEFAULT_LANE_CONFIG)
    ap.add_argument("--obstacles", type=Path, help="obstacles.csv to replay alongside a clip")
    ap.add_argument("--config", default=None, help="autonomy config.yaml (default: autonomy/config.yaml)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--duration", type=float, help="stop after this many seconds (default: end of clip)")
    ap.add_argument("--plot", type=Path, help="save a steer/throttle/state plot here")
    ap.add_argument("--show", action="store_true", help="open the plot window")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    perception, loop = build(args, cfg)

    try:
        if perception.fps is None:
            loop.run(duration_s=args.duration)
        else:
            # run in simulated time until the clip is used up (or --duration)
            limit = args.duration
            while not perception.exhausted and (limit is None or loop.clock() < limit):
                loop.run(n_ticks=1)
    except KeyboardInterrupt:
        pass

    history = loop.history
    period = 1.0 / cfg.loop.hz
    counts = Counter(h.state for h in history)
    lane_ticks = sum(1 for h in history if h.path_confidence > 0)
    print(f"{len(history)} ticks, {perception.frame_index + 1} frames, lane path on {lane_ticks} ticks")
    for s in State:
        if counts[s]:
            print(f"  {s.name:12s} {counts[s] * period:6.2f} s")
    prev = None
    for h in history:
        if h.state != prev:
            print(f"  t={h.t:7.2f}  -> {h.state.name}")
            prev = h.state
    if args.plot or args.show:
        from .replay import plot
        out = args.plot or DATA / "autonomy_run.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plot(history, out, args.show)
        print(f"plot: {out}")


if __name__ == "__main__":
    main()
