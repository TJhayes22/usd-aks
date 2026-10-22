"""Pure geometry helpers shared by the supervisor and controller."""
from typing import List, Sequence

import numpy as np

from .config import Config
from .types import Obstacle

# Classes that shape the path instead of blocking it.
BOUNDARY_CLASSES = ("cone",)


def stopping_distance(v: float, latency_s: float, decel_mps2: float) -> float:
    """Distance covered while reacting (v * latency) plus braking to zero (v^2 / 2a)."""
    v = max(v, 0.0)
    return v * latency_s + v * v / (2.0 * decel_mps2)


def stop_threshold(cls: str, v: float, cfg: Config) -> float:
    """Distance at which an obstacle of this class requires an urgent stop."""
    return stopping_distance(v, cfg.vehicle.latency_s, cfg.vehicle.brake_decel_mps2) + cfg.obstacles.margin(cls)


def corridor_polyline(path_points: np.ndarray, length_m: float) -> np.ndarray:
    """Path from the origin, extended straight along its last heading out to length_m.

    With no path, the corridor is a straight line ahead.
    """
    pts = np.asarray(path_points, dtype=float).reshape(-1, 2)
    origin = np.zeros((1, 2))
    if len(pts) == 0 or not np.allclose(pts[0], 0.0):
        pts = np.vstack([origin, pts])
    if len(pts) == 1:
        return np.array([[0.0, 0.0], [0.0, length_m]])
    direction = pts[-1] - pts[-2]
    norm = np.linalg.norm(direction)
    if norm > 0 and pts[-1][1] < length_m:
        pts = np.vstack([pts, pts[-1] + direction / norm * length_m])
    return pts


def distance_to_polyline(p: Sequence[float], polyline: np.ndarray) -> float:
    """Shortest distance from point p to a polyline of (N, 2) points."""
    p = np.asarray(p, dtype=float)
    a, b = polyline[:-1], polyline[1:]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    t = np.where(denom > 0, np.einsum("ij,ij->i", p - a, ab) / np.where(denom > 0, denom, 1.0), 0.0)
    closest = a + np.clip(t, 0.0, 1.0)[:, None] * ab
    return float(np.min(np.linalg.norm(closest - p, axis=1)))


def corridor_half_width(cfg: Config) -> float:
    return cfg.vehicle.vehicle_width_m / 2.0 + cfg.obstacles.corridor_margin_m


def relevant_obstacles(obstacles: List[Obstacle], path_points: np.ndarray, cfg: Config) -> List[Obstacle]:
    """Non-boundary obstacles ahead of the car and inside the corridor along the path, nearest first."""
    poly = corridor_polyline(path_points, cfg.obstacles.corridor_length_m)
    half = corridor_half_width(cfg)
    hits = [
        o for o in obstacles
        if o.cls not in BOUNDARY_CLASSES
        and 0.0 < o.y <= cfg.obstacles.corridor_length_m
        and distance_to_polyline((o.x, o.y), poly) <= half
    ]
    return sorted(hits, key=lambda o: o.y)
