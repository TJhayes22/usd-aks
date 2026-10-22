"""Fixed-rate loop: perception -> tracker -> planner -> supervisor -> controller -> safety -> vehicle.

Adapters must provide:
  perception.run(now) -> PerceptionFrame
  rc.run(now) -> RcInput
  vehicle.run(cmd, now) and vehicle.speed_mps
Clock and sleep are injectable so tests and replays can run in simulated time (SimClock).
"""
import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from .config import Config
from .controller import Controller
from .planner import Planner
from .safety import FULL_BRAKE, Safety
from .supervisor import Supervisor
from .tracker import Tracker
from .types import Command, State, World

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TickLog:
    t: float
    state: State
    cmd: Command
    speed_mps: float
    n_obstacles: int
    path_confidence: float


class SimClock:
    """Simulated time: sleep() advances the clock instantly."""

    def __init__(self, t0: float = 0.0):
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += max(dt, 0.0)


class Loop:
    def __init__(self, cfg: Config, perception, rc, vehicle,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.cfg = cfg
        self.perception, self.rc, self.vehicle = perception, rc, vehicle
        self.clock, self.sleep = clock, sleep
        self.tracker = Tracker()
        self.planner = Planner(cfg)
        self.supervisor = Supervisor(cfg)
        self.controller = Controller(cfg)
        self.safety = Safety(cfg)
        self.history: List[TickLog] = []
        self.overruns = 0

    def tick(self, now: float) -> TickLog:
        frame = self.perception.run(now)
        obstacles = self.tracker.run(frame.obstacles)
        path = self.planner.run(obstacles, frame.lane_center, frame.lane_confidence)
        world = World(t_perception=frame.t, obstacles=obstacles, path=path,
                      speed_mps=self.vehicle.speed_mps,
                      lane_center=frame.lane_center, lane_confidence=frame.lane_confidence)
        rc = self.rc.run(now)
        state = self.supervisor.run(world, rc, now)
        cmd = self.safety.run(state, self.controller.run(state, world), rc)
        self.vehicle.run(cmd, now)
        entry = TickLog(now, state, cmd, world.speed_mps, len(obstacles), path.confidence)
        self.history.append(entry)
        return entry

    def run(self, n_ticks: Optional[int] = None, duration_s: Optional[float] = None) -> List[TickLog]:
        """Run until n_ticks or duration_s (or forever). Any exception brakes the car before propagating."""
        period = 1.0 / self.cfg.loop.hz
        start = next_t = self.clock()
        ticks = 0
        try:
            while (n_ticks is None or ticks < n_ticks) and \
                    (duration_s is None or self.clock() - start < duration_s):
                self.tick(self.clock())
                ticks += 1
                next_t += period
                delay = next_t - self.clock()
                if delay > 0:
                    self.sleep(delay)
                else:
                    self.overruns += 1
                    log.warning("loop overrun by %.1f ms", -delay * 1000)
                    if -delay > period:
                        next_t = self.clock()   # too far behind: re-anchor instead of bursting
        except Exception:
            self.vehicle.run(FULL_BRAKE, self.clock())
            raise
        return self.history
