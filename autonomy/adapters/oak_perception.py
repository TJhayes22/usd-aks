"""Real OAK-D perception adapter (stub).

Plan: build the pipeline from oak-template/oak_obstacles.py, read ObstacleNode.latest each tick,
and convert to the car frame: x = LATERAL_SIGN * lateral_m, y = forward_m, with camera pitch and
height correction once the mount is measured. depthai is imported lazily so the rest of the
package never depends on it.
"""
from ..config import Config
from ..types import PerceptionFrame


class OakPerception:
    def __init__(self, cfg: Config, model_archive: str):
        import depthai  # noqa: F401  (only available in oak-env / on the Jetson)
        raise NotImplementedError("OakPerception is a stub; use MockPerception or CsvReplayPerception")

    def run(self, now: float) -> PerceptionFrame:
        raise NotImplementedError
