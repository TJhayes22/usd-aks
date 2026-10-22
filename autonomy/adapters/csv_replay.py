"""Replay obstacles.csv (written by oak-template/oak_obstacles.py) as perception frames.

CSV columns: ts,label,conf,forward_m,lateral_m. Rows with the same ts form one frame.
Timestamps are rebased so the first frame is t=0. Car frame: x = csv_lateral_sign * lateral_m,
y = forward_m (no pitch/height correction yet, same as perception).
Empty frames are not logged, so the newest frame is held and a long gap reads as stale data.
"""
import bisect
import csv
from collections import OrderedDict
from typing import List

from ..config import Config
from ..types import Obstacle, PerceptionFrame


class CsvReplayPerception:
    def __init__(self, path, cfg: Config):
        frames = OrderedDict()
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                frames.setdefault(float(row["ts"]), []).append(row)
        if not frames:
            raise ValueError(f"{path}: no rows")
        t0 = min(frames)
        sign = cfg.replay.csv_lateral_sign
        self.times: List[float] = []
        self.frames: List[List[Obstacle]] = []
        for ts in sorted(frames):
            t = ts - t0
            self.times.append(t)
            self.frames.append([
                Obstacle(cls=r["label"], x=sign * float(r["lateral_m"]), y=float(r["forward_m"]),
                         conf=float(r["conf"]), t=t)
                for r in frames[ts]
            ])

    @property
    def duration_s(self) -> float:
        return self.times[-1]

    def run(self, now: float) -> PerceptionFrame:
        i = bisect.bisect_right(self.times, now) - 1
        if i < 0:
            return PerceptionFrame(t=float("-inf"))
        return PerceptionFrame(t=self.times[i], obstacles=list(self.frames[i]))
