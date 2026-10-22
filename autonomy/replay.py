"""Run an obstacles.csv through the full stack in simulated time and plot steer / throttle / state.

The RC is scripted as: armed in MANUAL, mode switch flipped to AUTO when the first frame arrives.
The vehicle is MockVehicle, so speed is simulated, not measured.

    python -m autonomy.replay oak-template/obstacles.csv
    python -m autonomy.replay oak-template/obstacles.csv --out replay.png --show
"""
import argparse
from collections import Counter
from pathlib import Path

from .adapters.csv_replay import CsvReplayPerception
from .adapters.mock_rc import MockRc
from .adapters.mock_vehicle import MockVehicle
from .config import load_config
from .loop import Loop, SimClock
from .types import State

LEAD_IN_S = 0.5   # sim starts this long before the first frame so the car can arm in MANUAL


def replay(csv_path, cfg, tail_s: float):
    perception = CsvReplayPerception(csv_path, cfg)
    clock = SimClock(-LEAD_IN_S)
    rc = MockRc([(-LEAD_IN_S, {"arm": True, "mode_auto": False}), (0.0, {"mode_auto": True})])
    loop = Loop(cfg, perception, rc, MockVehicle(cfg), clock=clock, sleep=clock.sleep)
    return loop.run(duration_s=LEAD_IN_S + perception.duration_s + tail_s)


def plot(history, out_path, show: bool):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [h.t for h in history]
    order = list(State)
    fig, (ax_steer, ax_drive, ax_state) = plt.subplots(3, 1, sharex=True, figsize=(11, 7))

    ax_steer.plot(t, [h.cmd.steer_rad for h in history])
    ax_steer.set_ylabel("steer (rad)\n+ = right")
    ax_steer.grid(alpha=0.3)

    ax_drive.plot(t, [h.cmd.throttle for h in history], label="throttle")
    ax_drive.plot(t, [h.cmd.brake for h in history], label="brake")
    ax_drive.set_ylabel("command (0-1)")
    ax_drive.set_ylim(-0.05, 1.05)
    ax_drive.grid(alpha=0.3)
    ax_speed = ax_drive.twinx()
    ax_speed.plot(t, [h.speed_mps for h in history], "k--", lw=1, label="speed (sim)")
    ax_speed.set_ylabel("speed (m/s)")
    lines = ax_drive.get_legend_handles_labels()
    lines2 = ax_speed.get_legend_handles_labels()
    ax_drive.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="upper right", fontsize=8)

    ax_state.step(t, [order.index(h.state) for h in history], where="post")
    ax_state.set_yticks(range(len(order)))
    ax_state.set_yticklabels([s.name for s in order])
    ax_state.set_xlabel("time (s, from first frame)")
    ax_state.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    if show:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", help="obstacles.csv from oak_obstacles.py")
    ap.add_argument("--config", default=None, help="config.yaml (default: autonomy/config.yaml)")
    ap.add_argument("--out", default=None, help="PNG path (default: autonomy_replay.png next to the CSV)")
    ap.add_argument("--tail", type=float, default=1.0, help="seconds to keep running after the last frame")
    ap.add_argument("--show", action="store_true", help="also open the plot window")
    args = ap.parse_args()

    cfg = load_config(args.config)
    history = replay(args.csv, cfg, args.tail)
    out = Path(args.out) if args.out else Path(args.csv).with_name("autonomy_replay.png")
    plot(history, out, args.show)

    period = 1.0 / cfg.loop.hz
    counts = Counter(h.state for h in history)
    print(f"{len(history)} ticks at {cfg.loop.hz:g} Hz")
    for s in State:
        if counts[s]:
            print(f"  {s.name:12s} {counts[s] * period:6.2f} s")
    prev = None
    for h in history:
        if h.state != prev:
            print(f"  t={h.t:6.2f}  -> {h.state.name}")
            prev = h.state
    print(f"plot: {out}")


if __name__ == "__main__":
    main()
