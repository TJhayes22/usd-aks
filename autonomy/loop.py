"""Fixed-rate loop: perception -> tracker -> planner -> supervisor -> controller -> safety -> vehicle.

Adapters must provide:
  perception.run(now) -> PerceptionFrame
  rc.run(now) -> RcInput
  vehicle.run(cmd, now) and vehicle.speed_mps
  recorder.run(tick_log)                   (optional)
Any adapter may expose a `warnings` tuple (e.g. "recording stopped: disk cap"); the loop logs
changes and hands them to the supervisor's health status. Warnings never change the state.
Clock and sleep are injectable so tests and replays can run in simulated time (SimClock).
"""
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Tuple

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
    warnings: Tuple[str, ...] = ()


class SimClock:
    """Simulated time: sleep() advances the clock instantly."""

    def __init__(self, t0: float = 0.0):
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += max(dt, 0.0)


class Loop:
    def __init__(self, cfg: Config, perception, rc, vehicle, recorder=None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.cfg = cfg
        self.perception, self.rc, self.vehicle, self.recorder = perception, rc, vehicle, recorder
        self.clock, self.sleep = clock, sleep
        self.tracker = Tracker()
        self.planner = Planner(cfg)
        self.supervisor = Supervisor(cfg)
        self.controller = Controller(cfg)
        self.safety = Safety(cfg)
        # bounded so a long live run can't grow memory; the full record goes to the recorder
        self.history: Deque[TickLog] = deque(maxlen=cfg.loop.history_max_ticks)
        self.overruns = 0
        self._active_warnings: Tuple[str, ...] = ()

    def _collect_warnings(self) -> Tuple[str, ...]:
        parts = (self.perception, self.rc, self.vehicle, self.recorder)
        current = tuple(w for p in parts if p is not None for w in getattr(p, "warnings", ()))
        for w in current:
            if w not in self._active_warnings:
                log.warning("health: %s", w)
        for w in self._active_warnings:
            if w not in current:
                log.info("health cleared: %s", w)
        self._active_warnings = current
        return current

    def tick(self, now: float) -> TickLog:
        frame = self.perception.run(now)
        obstacles = self.tracker.run(frame.obstacles)
        path = self.planner.run(obstacles, frame.lane_path)
        world = World(t_perception=frame.t, obstacles=obstacles, path=path,
                      speed_mps=self.vehicle.speed_mps, lane_path=frame.lane_path)
        rc = self.rc.run(now)
        warnings = self._collect_warnings()
        state = self.supervisor.run(world, rc, now, warnings=warnings)
        cmd = self.safety.run(state, self.controller.run(state, world), rc)
        self.vehicle.run(cmd, now)
        entry = TickLog(now, state, cmd, world.speed_mps, len(obstacles), path.confidence, warnings)
        self.history.append(entry)
        if self.recorder is not None:
            self.recorder.run(entry)
        return entry

    def run(self, n_ticks: Optional[int] = None, duration_s: Optional[float] = None,
            until: Optional[Callable[[], bool]] = None) -> List[TickLog]:
        """Run until n_ticks, duration_s or until() (or forever).

        Anything that ends the loop abnormally (an exception, Ctrl+C, SystemExit from SIGTERM)
        sends full brake before propagating."""
        period = 1.0 / self.cfg.loop.hz
        start = next_t = self.clock()
        ticks = 0
        try:
            while (n_ticks is None or ticks < n_ticks) and \
                    (duration_s is None or self.clock() - start < duration_s) and \
                    (until is None or not until()):
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
        except BaseException:
            self.vehicle.run(FULL_BRAKE, self.clock())
            raise
        return list(self.history)
