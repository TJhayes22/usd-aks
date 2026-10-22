"""Pure pursuit steering plus speed and braking for AUTO / SLOW.

Steering: delta = atan(2 * L * sin(alpha) / Ld), with alpha the angle to the lookahead point
seen from the rear axle. Positive alpha / delta = right (+x).
Braking: stopping distance d_stop = v * latency + v^2 / (2 * a). Only obstacles inside the
corridor along the path count. person -> stop with a large margin; kart -> hold a following
distance; cones shape the path and are never braked for.
"""
import math

import numpy as np

from .config import Config
from .geometry import relevant_obstacles, stop_threshold
from .types import Command, Path, State, World

EPS = 1e-6


def lookahead_distance(v: float, cfg: Config) -> float:
    c = cfg.control
    return float(np.clip(c.k_ld_s * v + c.ld_min_m, c.ld_min_m, c.ld_max_m))


def lookahead_point(points: np.ndarray, ld: float) -> np.ndarray:
    """First point on the polyline (starting at the origin) at distance ld from the origin.

    If the polyline ends inside the circle it is extended along its last heading.
    """
    pts = np.vstack([np.zeros((1, 2)), points])
    for a, b in zip(pts[:-1], pts[1:]):
        if np.linalg.norm(b) >= ld:
            # solve |a + t(b - a)| = ld for t in [0, 1]
            d = b - a
            qa, qb, qc = d @ d, 2 * a @ d, a @ a - ld * ld
            t = (-qb + math.sqrt(max(qb * qb - 4 * qa * qc, 0.0))) / (2 * qa)
            return a + min(max(t, 0.0), 1.0) * d
    d = pts[-1] - pts[-2]
    norm = np.linalg.norm(d)
    if norm < EPS:
        return pts[-1]
    a = pts[-1]
    u = d / norm
    # extend: |a + s*u| = ld
    qb = 2 * a @ u
    s = (-qb + math.sqrt(max(qb * qb - 4 * (a @ a - ld * ld), 0.0))) / 2
    return a + s * u


def pure_pursuit(path: Path, v: float, cfg: Config) -> float:
    if len(path.points) == 0:
        return 0.0
    axle_pts = np.asarray(path.points, dtype=float) + [0.0, cfg.vehicle.camera_to_rear_axle_m]
    ld = lookahead_distance(v, cfg)
    target = lookahead_point(axle_pts, ld)
    alpha = math.atan2(target[0], target[1])
    return math.atan(2.0 * cfg.vehicle.wheelbase_m * math.sin(alpha) / ld)


def target_speed(state: State, cfg: Config) -> float:
    if state == State.AUTO:
        v = cfg.control.cruise_speed_mps
    elif state == State.SLOW:
        v = cfg.control.slow_speed_mps
    else:
        v = 0.0
    return min(v, cfg.vehicle.max_speed_mps)


def compute(state: State, world: World, cfg: Config) -> Command:
    """Command for AUTO / SLOW. Other states get a zero command; safety.py forces the brake."""
    if state not in (State.AUTO, State.SLOW):
        return Command()

    v = max(world.speed_mps, 0.0)
    v_target = target_speed(state, cfg)
    brake = 0.0

    for o in relevant_obstacles(world.obstacles, world.path.points, cfg):
        if o.cls == "kart":
            gap_limit = cfg.obstacles.k_follow_per_s * (o.y - cfg.obstacles.follow_distance_m)
            v_target = min(v_target, max(gap_limit, 0.0))
        if o.y <= stop_threshold(o.cls, v, cfg):
            v_target = 0.0
            room = o.y - v * cfg.vehicle.latency_s - cfg.obstacles.margin(o.cls)
            needed = 1.0 if room <= 0 else (v * v / (2.0 * max(room, EPS))) / cfg.vehicle.brake_decel_mps2
            brake = max(brake, float(np.clip(needed, 0.0, 1.0)))

    if v > v_target + cfg.control.speed_tolerance_mps:
        brake = max(brake, float(np.clip(cfg.control.kp_speed * (v - v_target), 0.0, 1.0)))

    throttle = 0.0
    if brake == 0.0 and v_target > 0.0:
        ff = v_target / cfg.vehicle.max_speed_mps
        throttle = float(np.clip(ff + cfg.control.kp_speed * (v_target - v), 0.0, cfg.control.max_throttle))

    return Command(steer_rad=pure_pursuit(world.path, v, cfg), throttle=throttle, brake=brake)


class Controller:
    """Donkey-style part: run(state, world) -> Command."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, state: State, world: World) -> Command:
        return compute(state, world, self.cfg)
