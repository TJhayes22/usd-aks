"""Cones or a lane center line -> a center path in the car frame.

Cones: split left (x < 0) / right (x >= 0), sort each side by distance ahead, pair by rank,
and take midpoints. Cones without a partner (or a whole missing side) are offset half the
course width toward the center.

Priority: cones on both sides > lane center line > one-sided cones > nothing (confidence 0).
"""
from typing import List, Optional, Sequence

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


def lane_path_from_poly(coeffs: Optional[Sequence[float]], confidence: float, cfg: Config) -> Path:
    """Lane center fit x = a*y^2 + b*y + c (meters, car frame; lane_detection's LaneResult.center)
    sampled from lane_sample_start_m out to the controller's longest lookahead."""
    if coeffs is None or confidence <= 0:
        return Path()
    p = cfg.planner
    ys = np.arange(p.lane_sample_start_m, cfg.control.ld_max_m + p.lane_sample_step_m / 2, p.lane_sample_step_m)
    xs = np.polyval(np.asarray(coeffs, dtype=float), ys)
    return Path(np.column_stack([xs, ys]), float(confidence), "lanes")


def plan(obstacles: List[Obstacle], cfg: Config, lane_path: Optional[Path] = None) -> Path:
    cones = cone_path(obstacles, cfg)
    if cones.source == "cones":                      # cones on both sides
        return cones
    if lane_path is not None and lane_path.confidence > 0 and len(lane_path.points):
        return lane_path
    return cones                                     # one-sided cones, or nothing


class Planner:
    """Donkey-style part: run(obstacles, lane_path) -> Path."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, obstacles: List[Obstacle], lane_path: Optional[Path] = None) -> Path:
        return plan(obstacles, self.cfg, lane_path)
