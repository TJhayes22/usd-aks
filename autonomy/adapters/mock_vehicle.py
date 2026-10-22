"""Mock vehicle: logs every command and integrates a crude speed model for feedback."""
from typing import List, Optional, Tuple

from ..config import Config
from ..types import Command


class MockVehicle:
    def __init__(self, cfg: Config, speed_mps: float = 0.0):
        self.cfg = cfg
        self.speed_mps = speed_mps
        self.log: List[Tuple[float, Command]] = []
        self._last_t: Optional[float] = None

    def run(self, cmd: Command, now: float) -> None:
        if self._last_t is not None:
            dt = max(now - self._last_t, 0.0)
            # first-order drive with drag: throttle t settles at t * max_speed
            v = self.cfg.vehicle
            drive = v.accel_max_mps2 * (cmd.throttle - self.speed_mps / v.max_speed_mps)
            accel = drive - cmd.brake * v.brake_decel_mps2
            self.speed_mps = min(max(self.speed_mps + accel * dt, 0.0), self.cfg.vehicle.max_speed_mps)
        self._last_t = now
        self.log.append((now, cmd))
