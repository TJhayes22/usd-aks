import pytest

from autonomy.safety import apply
from autonomy.types import Command, State

from .conftest import rc_ok


@pytest.mark.parametrize("state", [State.URGENT_STOP, State.FAULT, State.DISARMED])
def test_forced_brake(cfg, state):
    out = apply(state, Command(0.3, 1.0, 0.0), rc_ok(0, steer=0.2, throttle=1.0), cfg)
    assert out == Command(0.0, 0.0, 1.0)


def test_clamps_steer_and_throttle(cfg):
    out = apply(State.AUTO, Command(5.0, 7.0, 0.0), rc_ok(0), cfg)
    assert out.steer_rad == cfg.vehicle.max_steer_rad
    assert out.throttle == cfg.control.max_throttle
    out = apply(State.AUTO, Command(-5.0, -1.0, 3.0), rc_ok(0), cfg)
    assert out.steer_rad == -cfg.vehicle.max_steer_rad
    assert (out.throttle, out.brake) == (0.0, 1.0)


def test_brake_wins_over_throttle(cfg):
    out = apply(State.SLOW, Command(0.0, 0.5, 0.2), rc_ok(0), cfg)
    assert (out.throttle, out.brake) == (0.0, 0.2)


def test_manual_passes_rc_through(cfg):
    out = apply(State.MANUAL, Command(0.3, 1.0, 0.0), rc_ok(0, steer=-0.1, throttle=0.4), cfg)
    assert out == Command(-0.1, 0.4, 0.0)
    out = apply(State.MANUAL, Command(), rc_ok(0, steer=9.0, throttle=-0.5), cfg)
    assert out == Command(cfg.vehicle.max_steer_rad, 0.0, 0.5)
