"""Tracker stub. Later: associate detections across frames, smooth positions, estimate velocity."""
from typing import List

from .types import Obstacle


class Tracker:
    """Donkey-style part: run(obstacles) -> obstacles. Currently a pass-through."""

    def run(self, obstacles: List[Obstacle]) -> List[Obstacle]:
        return list(obstacles)
