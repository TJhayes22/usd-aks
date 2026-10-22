import numpy as np
import pytest

from autonomy.config import load_config
from autonomy.types import Obstacle, Path, RcInput, World


@pytest.fixture
def cfg():
    return load_config()


def rc_ok(now, mode_auto=True, **kw):
    """RC input with a healthy, fresh link."""
    return RcInput(t=now, link_ok=True, mode_auto=mode_auto, **kw)


def straight_path(length=5.0, confidence=1.0):
    ys = np.linspace(0.0, length, 11)
    return Path(np.column_stack([np.zeros_like(ys), ys]), confidence, "cones")


def world(now, obstacles=(), path=None, speed=0.0):
    return World(t_perception=now, obstacles=list(obstacles),
                 path=path if path is not None else straight_path(), speed_mps=speed)


def obs(cls, x, y, t=0.0, conf=0.9):
    return Obstacle(cls=cls, x=x, y=y, conf=conf, t=t)
