import math

import numpy as np
import pytest

from autonomy.controller import compute, lookahead_point, pure_pursuit
from autonomy.geometry import corridor_half_width, stop_threshold
from autonomy.planner import plan
from autonomy.types import Path, State

from .conftest import obs, straight_path, world

NOW = 50.0


def test_cone_gate_steers_toward_midpoint(cfg):
    right_gate = [obs("cone", -0.1, 2.0), obs("cone", 1.1, 2.0)]   # midpoint x = +0.5
    left_gate = [obs("cone", -1.1, 2.0), obs("cone", 0.1, 2.0)]    # midpoint x = -0.5
    v = cfg.control.slow_speed_mps
    right = compute(State.AUTO, world(NOW, right_gate, plan(right_gate, cfg), speed=v), cfg)
    left = compute(State.AUTO, world(NOW, left_gate, plan(left_gate, cfg), speed=v), cfg)
    assert right.steer_rad > 0 > left.steer_rad
    assert right.steer_rad == pytest.approx(-left.steer_rad)


def test_pure_pursuit_formula(cfg):
    path = Path(np.array([[0.0, 0.0], [1.0, 5.0]]), 1.0, "lane")
    v = 1.0
    ld = cfg.control.k_ld_s * v + cfg.control.ld_min_m
    tgt = lookahead_point(path.points + [0.0, cfg.vehicle.camera_to_rear_axle_m], ld)
    assert np.linalg.norm(tgt) == pytest.approx(ld)
    alpha = math.atan2(tgt[0], tgt[1])
    expected = math.atan(2 * cfg.vehicle.wheelbase_m * math.sin(alpha) / ld)
    assert pure_pursuit(path, v, cfg) == pytest.approx(expected)


def test_lookahead_extends_short_path():
    pts = np.array([[0.0, 0.5], [0.0, 1.0]])
    np.testing.assert_allclose(lookahead_point(pts, 3.0), [0.0, 3.0])


def test_straight_path_zero_steer(cfg):
    cmd = compute(State.AUTO, world(NOW, speed=1.0), cfg)
    assert cmd.steer_rad == pytest.approx(0.0)
    assert cmd.throttle > 0 and cmd.brake == 0


def test_person_in_corridor_brakes(cfg):
    v = cfg.control.cruise_speed_mps
    cmd = compute(State.AUTO, world(NOW, [obs("person", 0.0, 2.5)], speed=v), cfg)
    assert cmd.brake > 0 and cmd.throttle == 0


def test_person_right_at_car_full_brake(cfg):
    cmd = compute(State.SLOW, world(NOW, [obs("person", 0.0, 0.5)], speed=1.0), cfg)
    assert cmd.brake == 1.0


def test_person_outside_corridor_ignored(cfg):
    x = corridor_half_width(cfg) + 0.5
    v = cfg.control.cruise_speed_mps
    assert stop_threshold("person", v, cfg) > 2.5
    cmd = compute(State.AUTO, world(NOW, [obs("person", x, 2.5)], speed=v), cfg)
    assert cmd.brake == 0 and cmd.throttle > 0


def test_kart_ahead_keeps_following_distance(cfg):
    gap = cfg.obstacles.follow_distance_m + 0.5
    limit = cfg.obstacles.k_follow_per_s * 0.5
    fast = compute(State.AUTO, world(NOW, [obs("kart", 0.0, gap)], speed=cfg.control.cruise_speed_mps), cfg)
    assert fast.brake > 0 and fast.throttle == 0          # too fast for the gap: slow down
    matched = compute(State.AUTO, world(NOW, [obs("kart", 0.0, gap)], speed=limit), cfg)
    assert matched.brake == 0
    assert matched.throttle == pytest.approx(limit / cfg.vehicle.max_speed_mps)
    close = compute(State.AUTO, world(NOW, [obs("kart", 0.0, cfg.obstacles.follow_distance_m - 0.2)], speed=0.0), cfg)
    assert close.throttle == 0                            # inside following distance: don't move up


def test_cone_in_path_not_braked_for(cfg):
    cmd = compute(State.AUTO, world(NOW, [obs("cone", 0.0, 0.8)], speed=cfg.control.cruise_speed_mps), cfg)
    assert cmd.brake == 0


def test_slow_state_lower_speed(cfg):
    auto = compute(State.AUTO, world(NOW, speed=0.0), cfg)
    slow = compute(State.SLOW, world(NOW, speed=0.0), cfg)
    assert auto.throttle > slow.throttle > 0


def test_non_driving_states_zero_command(cfg):
    for s in (State.DISARMED, State.MANUAL, State.URGENT_STOP, State.FAULT):
        cmd = compute(s, world(NOW, path=straight_path(), speed=1.0), cfg)
        assert (cmd.steer_rad, cmd.throttle) == (0.0, 0.0)
