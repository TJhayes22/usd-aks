"""Cones or a lane center line -> a center path in the car frame.

Cones: split left (x < 0) / right (x >= 0), sort each side by distance ahead, pair by rank,
and take midpoints. Cones without a partner (or a whole missing side) are offset half the
course width toward the center. No cones and no lane -> confidence 0.
"""
from typing import List, Optional

import numpy as np

from .config import Config
from .types import Obstacle, Path

ORIGIN = np.zeros((1, 2))


def _xy(cones: List[Obstacle]) -> np.ndarray:
    return np.array([[c.x, c.y] for c in sorted(cones, key=lambda c: c.y)]).reshape(-1, 2)


def _with_origin(points: np.ndarray) -> np.ndarray:
    return np.vstack([ORIGIN, points[np.argsort(points[:, 1], kind="stable")]])


def cone_path(obstacles: List[Obstacle], cfg: Config) -> Path:
    cones = [o for o in obstacles if o.cls == "cone" and o.y > 0]
    left = _xy([c for c in cones if c.x < 0])
    right = _xy([c for c in cones if c.x >= 0])
    half = cfg.planner.cone_track_width_m / 2.0
    n = min(len(left), len(right))

    if n == 0 and len(left) == 0 and len(right) == 0:
        return Path()
    if n == 0:
        if len(left):
            return Path(_with_origin(left + [half, 0.0]), cfg.planner.one_side_confidence, "cones_left")
        return Path(_with_origin(right - [half, 0.0]), cfg.planner.one_side_confidence, "cones_right")

    mids = (left[:n] + right[:n]) / 2.0
    extra = np.vstack([left[n:] + [half, 0.0], right[n:] - [half, 0.0]])
    confidence = min(1.0, n / cfg.planner.full_confidence_pairs)
    return Path(_with_origin(np.vstack([mids, extra])), confidence, "cones")


def lane_path(lane_center: Optional[np.ndarray], lane_confidence: float) -> Path:
    if lane_center is None or len(lane_center) < 2 or lane_confidence <= 0:
        return Path()
    pts = np.asarray(lane_center, dtype=float).reshape(-1, 2)
    return Path(pts[np.argsort(pts[:, 1], kind="stable")], float(lane_confidence), "lane")


def plan(obstacles: List[Obstacle], cfg: Config,
         lane_center: Optional[np.ndarray] = None, lane_confidence: float = 0.0) -> Path:
    cones = cone_path(obstacles, cfg)
    lane = lane_path(lane_center, lane_confidence)
    first, second = (lane, cones) if cfg.planner.prefer == "lane" else (cones, lane)
    if first.confidence > 0:
        return first
    return second


class Planner:
    """Donkey-style part: run(obstacles, lane_center, lane_confidence) -> Path."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, obstacles: List[Obstacle], lane_center: Optional[np.ndarray] = None,
            lane_confidence: float = 0.0) -> Path:
        return plan(obstacles, self.cfg, lane_center, lane_confidence)
