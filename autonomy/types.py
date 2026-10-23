"""Shared data types. Car frame: meters, x right, y forward, origin on the ground at the camera."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class State(Enum):
    DISARMED = "DISARMED"
    MANUAL = "MANUAL"
    AUTO = "AUTO"
    SLOW = "SLOW"
    URGENT_STOP = "URGENT_STOP"
    FAULT = "FAULT"


@dataclass(frozen=True)
class Obstacle:
    cls: str        # "cone" / "person" / "kart"
    x: float        # m, right of the camera
    y: float        # m, ahead of the camera
    conf: float
    t: float        # s, perception timestamp
    box: Optional[Tuple[float, float, float, float]] = None   # detector box (xmin, ymin, xmax, ymax), 0-1 image coords


def _empty_points() -> np.ndarray:
    return np.zeros((0, 2))


@dataclass
class Path:
    points: np.ndarray = field(default_factory=_empty_points)   # (N, 2) of (x, y), nearest first
    confidence: float = 0.0                                      # 0 = no usable path
    source: str = "none"                                         # cones / cones_left / cones_right / lanes / none


@dataclass(frozen=True)
class Command:
    steer_rad: float = 0.0   # positive = steer right (+x)
    throttle: float = 0.0    # 0..1
    brake: float = 0.0       # 0..1


@dataclass(frozen=True)
class RcInput:
    t: float                   # s, time the last RC packet arrived
    link_ok: bool              # receiver reports a live link
    mode_auto: bool = False    # mode switch: True = AUTO requested, False = MANUAL
    arm: bool = False
    reset: bool = False        # clears a latched URGENT_STOP (with mode switch on MANUAL)
    estop: bool = False        # software Urgent Stop from the transmitter
    steer: float = 0.0         # rad, manual steering
    throttle: float = 0.0      # -1..1, manual throttle; negative = brake
    allowed_auto: Optional[bool] = None   # RCS Black Box "allowed control mode"; None = no black box


@dataclass
class PerceptionFrame:
    """What a perception adapter returns each tick."""
    t: float                                   # s, timestamp of this frame; -inf = no frame yet
    obstacles: List[Obstacle] = field(default_factory=list)
    lane_path: Optional[Path] = None           # lane center line as a path, if lane detection ran


@dataclass
class World:
    t_perception: float                        # s, timestamp of the newest perception frame
    obstacles: List[Obstacle] = field(default_factory=list)
    path: Path = field(default_factory=Path)
    speed_mps: float = 0.0                     # current speed estimate
    lane_path: Optional[Path] = None           # lane center line, if lane detection ran
