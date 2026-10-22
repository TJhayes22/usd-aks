"""Last stage before the actuators: clamp to limits and force the brake when not allowed to drive.

URGENT_STOP / FAULT / DISARMED: zero throttle, full brake, wheels straight.
MANUAL: RC sticks pass through (negative throttle = brake), still clamped.
Throttle and brake are never both non-zero; brake wins.
"""
import numpy as np

from .config import Config
from .types import Command, RcInput, State

STOPPED = (State.URGENT_STOP, State.FAULT, State.DISARMED)
FULL_BRAKE = Command(steer_rad=0.0, throttle=0.0, brake=1.0)


def clamp(cmd: Command, cfg: Config) -> Command:
    lim = cfg.vehicle.max_steer_rad
    steer = float(np.clip(cmd.steer_rad, -lim, lim))
    brake = float(np.clip(cmd.brake, 0.0, 1.0))
    throttle = 0.0 if brake > 0 else float(np.clip(cmd.throttle, 0.0, cfg.control.max_throttle))
    return Command(steer_rad=steer, throttle=throttle, brake=brake)


def apply(state: State, cmd: Command, rc: RcInput, cfg: Config) -> Command:
    if state in STOPPED:
        return FULL_BRAKE
    if state == State.MANUAL:
        cmd = Command(steer_rad=rc.steer, throttle=max(rc.throttle, 0.0), brake=abs(min(rc.throttle, 0.0)))
    return clamp(cmd, cfg)


class Safety:
    """Donkey-style part: run(state, cmd, rc) -> Command."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, state: State, cmd: Command, rc: RcInput) -> Command:
        return apply(state, cmd, rc, self.cfg)
