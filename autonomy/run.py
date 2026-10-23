"""Run the stack with lane detection as the path source, from a recorded clip or the OAK-D.

    python -m autonomy.run --source clip.mp4                          # offline, simulated time
    python -m autonomy.run --source clip.mp4 --obstacles obstacles.csv --plot run.png
    python -m autonomy.run --source oak                               # live camera, real time (Ctrl+C stops)

RC and vehicle are MOCKS: the mock RC arms and switches to AUTO by itself, and nothing moves.
That auto-arm must go when the real RC receiver and motor adapters replace the mocks.
--source oak opens the camera for lanes only; it can't share the OAK-D with oak_obstacles.py.

Logs go to $AKS_DATA/logs/aks.log (rotating, size-capped) and every tick to
$AKS_DATA/logs/ticks_<time>.csv until the disk budget (AKS_DATA_MAX_GB / AKS_DATA_MIN_FREE_GB)
runs out; then tick recording stops with a warning and driving carries on.
"""
import argparse
import logging
import signal
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from .adapters.csv_replay import CsvReplayPerception
from .adapters.lane_perception import LanePerception
from .adapters.mock_rc import MockRc
from .adapters.mock_vehicle import MockVehicle
from .adapters.tick_recorder import TickRecorder
from .config import load_config
from .loop import Loop, SimClock
from .paths import DATA, REPO
from .storage import DataBudget, setup_logging
from .types import State

DEFAULT_LANE_CONFIG = REPO / "lane_detection" / "lane_config.json"
LEAD_IN_S = 0.5   # time before the first frame, so the car can arm in MANUAL

log = logging.getLogger(__name__)


def _sigterm(*_):
    raise SystemExit("SIGTERM")    # systemctl stop -> SystemExit -> loop sends full brake


def build(args, cfg, recorder):
    obstacles = CsvReplayPerception(args.obstacles, cfg) if args.obstacles else None
    perception = LanePerception.from_source(cfg, args.source, args.lane_config,
                                            args.width, args.height, obstacle_source=obstacles)
    if perception.fps is None:   # live: real time
        t0 = time.monotonic()
        rc = MockRc([(t0, {"arm": True, "mode_auto": False}), (t0 + LEAD_IN_S, {"mode_auto": True})])
        loop = Loop(cfg, perception, rc, MockVehicle(cfg), recorder=recorder)
    else:                        # recorded clip: simulated time at the clip's own speed
        clock = SimClock(-LEAD_IN_S)
        rc = MockRc([(-LEAD_IN_S, {"arm": True, "mode_auto": False}), (0.0, {"mode_auto": True})])
        loop = Loop(cfg, perception, rc, MockVehicle(cfg), recorder=recorder, clock=clock, sleep=clock.sleep)
    return perception, loop


def summarize(loop, perception, cfg):
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
    health = loop.supervisor.health
    print(f"health: {'ok' if health.ok else 'NOT ok'} ({health.state.name})"
          + "".join(f"\n  warning: {w}" for w in health.warnings))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True, help="'oak', a webcam index, or a video/image file")
    ap.add_argument("--lane-config", type=Path, default=DEFAULT_LANE_CONFIG)
    ap.add_argument("--obstacles", type=Path, help="obstacles.csv to replay alongside a clip")
    ap.add_argument("--config", default=None, help="autonomy config.yaml (default: autonomy/config.yaml)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--duration", type=float, help="stop after this many seconds (default: end of clip)")
    ap.add_argument("--no-record", action="store_true", help="don't write the per-tick CSV")
    ap.add_argument("--plot", type=Path, help="save a steer/throttle/state plot here")
    ap.add_argument("--show", action="store_true", help="open the plot window")
    args = ap.parse_args()

    log_file = setup_logging(DATA / "logs")
    signal.signal(signal.SIGTERM, _sigterm)
    cfg = load_config(args.config)
    recorder = None
    if not args.no_record:
        recorder = TickRecorder(DATA / "logs" / f"ticks_{datetime.now():%Y%m%d_%H%M%S}.csv", DataBudget(DATA))
    perception, loop = build(args, cfg, recorder)
    log.info("source=%s lane_config=%s log=%s", args.source, args.lane_config, log_file)

    try:
        until = None if perception.fps is None else (lambda: perception.exhausted)
        loop.run(duration_s=args.duration, until=until)
    except (KeyboardInterrupt, SystemExit):
        log.info("stopped; full brake sent")
    finally:
        if recorder is not None:
            recorder.close()

    summarize(loop, perception, cfg)
    if recorder is not None:
        stopped = f", STOPPED: {recorder.stopped_reason}" if recorder.stopped_reason else ""
        print(f"ticks: {recorder.path} ({recorder.rows} rows{stopped})")
    if args.plot or args.show:
        from .replay import plot
        out = args.plot or DATA / "autonomy_run.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plot(list(loop.history), out, args.show)
        print(f"plot: {out}")


if __name__ == "__main__":
    main()
