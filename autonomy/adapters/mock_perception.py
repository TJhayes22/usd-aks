"""Scripted perception for tests and bench runs.

A script is a list of (t_start, obstacles) segments in loop time. obstacles is a list of
(cls, x, y) tuples, or None to simulate the camera going silent (frames stop, data goes stale).
"""
from typing import List, Optional, Sequence, Tuple

from ..types import Obstacle, PerceptionFrame

Segment = Tuple[float, Optional[Sequence[Tuple[str, float, float]]]]


class MockPerception:
    def __init__(self, script: List[Segment], conf: float = 0.9):
        self.script = sorted(script, key=lambda s: s[0])
        self.conf = conf
        self.last_t = float("-inf")

    def run(self, now: float) -> PerceptionFrame:
        active = [s for s in self.script if s[0] <= now]
        if not active or active[-1][1] is None:
            return PerceptionFrame(t=self.last_t)   # silent camera: timestamp stops advancing
        self.last_t = now
        obstacles = [Obstacle(cls, x, y, self.conf, now) for cls, x, y in active[-1][1]]
        return PerceptionFrame(t=now, obstacles=obstacles)


def cone_gates(n: int = 4, spacing: float = 1.5, half_width: float = 0.5, x_offset: float = 0.0):
    """Pairs of cones straight ahead, as (cls, x, y) tuples."""
    out = []
    for i in range(1, n + 1):
        y = i * spacing
        out += [("cone", x_offset - half_width, y), ("cone", x_offset + half_width, y)]
    return out
