"""Mode state machine. step() is pure; Supervisor wraps it with the little memory it needs.

Priority order (first match wins):
  1. RC link lost / RC timeout / RC e-stop / software stop request -> URGENT_STOP (rulebook)
  2. URGENT_STOP is latched until: link OK + mode switch on MANUAL + reset -> MANUAL
  3. Stale perception while driving autonomously -> FAULT; FAULT held for fault_escalate_s -> URGENT_STOP
  4. MANUAL <-> AUTO only on the RC mode switch (AUTO also needs fresh perception)
  5. In AUTO/SLOW: obstacle inside stop distance -> URGENT_STOP; inside slow_factor x -> SLOW;
     path confidence low for low_confidence_ticks in a row -> SLOW; otherwise AUTO
"""
from typing import Optional, Sequence

from .config import Config
from .geometry import relevant_obstacles, stop_threshold
from .types import Health, RcInput, State, World

DRIVING = (State.AUTO, State.SLOW, State.FAULT)


def rc_lost(rc: RcInput, now: float, cfg: Config) -> bool:
    return (not rc.link_ok) or (now - rc.t > cfg.timeouts.rc_timeout_s) or rc.estop


def perception_stale(world: World, now: float, cfg: Config) -> bool:
    return now - world.t_perception > cfg.timeouts.perception_timeout_s


def path_confidence_low(world: World, cfg: Config) -> bool:
    return world.path.confidence < cfg.supervisor.min_path_confidence


def assess(world: World, cfg: Config, low_conf_ticks: int = 1) -> State:
    """AUTO, SLOW or URGENT_STOP from obstacles in the corridor and path confidence.

    low_conf_ticks = consecutive ticks (including this one) the path confidence has been low."""
    slow = path_confidence_low(world, cfg) and low_conf_ticks >= cfg.supervisor.low_confidence_ticks
    for o in relevant_obstacles(world.obstacles, world.path.points, cfg):
        threshold = stop_threshold(o.cls, world.speed_mps, cfg)
        if o.y <= threshold:
            return State.URGENT_STOP
        if o.y <= cfg.supervisor.slow_factor * threshold:
            slow = True
    return State.SLOW if slow else State.AUTO


def step(state: State, world: World, rc: RcInput, now: float, cfg: Config,
         fault_since: Optional[float] = None, stop_request: bool = False,
         low_conf_ticks: int = 1) -> State:
    """Next state. fault_since is when the current FAULT began (None if not in FAULT).
    low_conf_ticks counts consecutive low-path-confidence ticks, this one included."""
    if state == State.DISARMED:
        if not rc_lost(rc, now, cfg) and rc.arm and not rc.mode_auto:
            return State.MANUAL
        return State.DISARMED

    if stop_request or rc_lost(rc, now, cfg):
        return State.URGENT_STOP

    if state == State.URGENT_STOP:
        return State.MANUAL if (rc.reset and not rc.mode_auto) else State.URGENT_STOP

    if not rc.mode_auto:
        return State.MANUAL   # driver takes over, including from FAULT

    stale = perception_stale(world, now, cfg)
    if state == State.MANUAL:
        if stale or rc.allowed_auto is False:
            return State.MANUAL
        return assess(world, cfg, low_conf_ticks)

    # AUTO, SLOW or FAULT with the switch on AUTO
    if rc.allowed_auto is False:
        return State.URGENT_STOP   # race control revoked autonomy mid-run
    if stale:
        if state == State.FAULT and fault_since is not None \
                and now - fault_since >= cfg.timeouts.fault_escalate_s:
            return State.URGENT_STOP
        return State.FAULT
    return assess(world, cfg, low_conf_ticks)


class Supervisor:
    """Donkey-style part: run(world, rc, now) -> State."""

    def __init__(self, cfg: Config, initial: State = State.DISARMED):
        self.cfg = cfg
        self.state = initial
        self.fault_since: Optional[float] = None
        self.low_conf_ticks = 0
        self.health = Health(initial, ok=True)

    def run(self, world: World, rc: RcInput, now: float, stop_request: bool = False,
            warnings: Sequence[str] = ()) -> State:
        """warnings (e.g. recording stopped) are reported in self.health; they never change the state."""
        self.low_conf_ticks = self.low_conf_ticks + 1 if path_confidence_low(world, self.cfg) else 0
        new = step(self.state, world, rc, now, self.cfg, self.fault_since, stop_request, self.low_conf_ticks)
        self.health = Health(new, ok=not warnings and new not in (State.FAULT, State.URGENT_STOP),
                             warnings=tuple(warnings))
        if new == State.FAULT and self.state != State.FAULT:
            self.fault_since = now
        elif new != State.FAULT:
            self.fault_since = None
        self.state = new
        return new
