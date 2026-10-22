import numpy as np

from autonomy.geometry import stop_threshold
from autonomy.supervisor import Supervisor, step
from autonomy.types import Path, RcInput, State

from .conftest import obs, rc_ok, world

NOW = 100.0


def test_person_3m_ahead_urgent_stop(cfg):
    v = cfg.control.cruise_speed_mps
    assert stop_threshold("person", v, cfg) >= 3.0, "config no longer stops for a person at 3 m at cruise"
    w = world(NOW, [obs("person", 0.0, 3.0)], speed=v)
    assert step(State.AUTO, w, rc_ok(NOW), NOW, cfg) == State.URGENT_STOP


def test_person_within_twice_stop_distance_slows(cfg):
    v = cfg.control.cruise_speed_mps
    y = 1.5 * stop_threshold("person", v, cfg)
    w = world(NOW, [obs("person", 0.0, y)], speed=v)
    assert step(State.AUTO, w, rc_ok(NOW), NOW, cfg) == State.SLOW


def test_person_outside_corridor_ignored(cfg):
    w = world(NOW, [obs("person", 3.0, 2.0)], speed=cfg.control.cruise_speed_mps)
    assert step(State.AUTO, w, rc_ok(NOW), NOW, cfg) == State.AUTO


def test_cone_in_path_is_not_an_obstacle(cfg):
    w = world(NOW, [obs("cone", 0.0, 0.5)], speed=cfg.control.cruise_speed_mps)
    assert step(State.AUTO, w, rc_ok(NOW), NOW, cfg) == State.AUTO


def test_empty_frame_slows(cfg):
    w = world(NOW, [], path=Path(np.zeros((0, 2)), 0.0, "none"))
    assert step(State.AUTO, w, rc_ok(NOW), NOW, cfg) == State.SLOW


def test_stale_perception_faults_then_escalates(cfg):
    sup = Supervisor(cfg, initial=State.AUTO)
    stale_t = NOW - cfg.timeouts.perception_timeout_s - 0.1
    assert sup.run(world(stale_t), rc_ok(NOW), NOW) == State.FAULT
    later = NOW + cfg.timeouts.fault_escalate_s
    assert sup.run(world(stale_t), rc_ok(later), later) == State.URGENT_STOP


def test_fault_recovers_if_perception_returns_quickly(cfg):
    sup = Supervisor(cfg, initial=State.AUTO)
    stale_t = NOW - cfg.timeouts.perception_timeout_s - 0.1
    assert sup.run(world(stale_t), rc_ok(NOW), NOW) == State.FAULT
    t = NOW + cfg.timeouts.fault_escalate_s / 2
    assert sup.run(world(t), rc_ok(t), t) == State.AUTO


def test_stale_perception_ignored_in_manual(cfg):
    w = world(NOW - 10.0)
    assert step(State.MANUAL, w, rc_ok(NOW, mode_auto=False), NOW, cfg) == State.MANUAL


def test_rc_loss_urgent_stop(cfg):
    lost = RcInput(t=NOW, link_ok=False, mode_auto=True)
    for s in (State.MANUAL, State.AUTO, State.SLOW, State.FAULT):
        assert step(s, world(NOW), lost, NOW, cfg) == State.URGENT_STOP


def test_rc_timeout_urgent_stop(cfg):
    old = rc_ok(NOW - cfg.timeouts.rc_timeout_s - 0.1)
    assert step(State.AUTO, world(NOW), old, NOW, cfg) == State.URGENT_STOP


def test_rc_estop_and_software_stop(cfg):
    assert step(State.AUTO, world(NOW), rc_ok(NOW, estop=True), NOW, cfg) == State.URGENT_STOP
    assert step(State.AUTO, world(NOW), rc_ok(NOW), NOW, cfg, stop_request=True) == State.URGENT_STOP


def test_urgent_stop_latched_until_reset_in_manual(cfg):
    w = world(NOW)
    assert step(State.URGENT_STOP, w, rc_ok(NOW, mode_auto=True), NOW, cfg) == State.URGENT_STOP
    assert step(State.URGENT_STOP, w, rc_ok(NOW, mode_auto=True, reset=True), NOW, cfg) == State.URGENT_STOP
    assert step(State.URGENT_STOP, w, rc_ok(NOW, mode_auto=False), NOW, cfg) == State.URGENT_STOP
    assert step(State.URGENT_STOP, w, rc_ok(NOW, mode_auto=False, reset=True), NOW, cfg) == State.MANUAL


def test_manual_auto_only_on_switch(cfg):
    w = world(NOW)
    assert step(State.MANUAL, w, rc_ok(NOW, mode_auto=False), NOW, cfg) == State.MANUAL
    assert step(State.MANUAL, w, rc_ok(NOW, mode_auto=True), NOW, cfg) == State.AUTO
    assert step(State.AUTO, w, rc_ok(NOW, mode_auto=False), NOW, cfg) == State.MANUAL


def test_arming(cfg):
    w = world(NOW)
    assert step(State.DISARMED, w, rc_ok(NOW, mode_auto=False), NOW, cfg) == State.DISARMED
    assert step(State.DISARMED, w, rc_ok(NOW, mode_auto=True, arm=True), NOW, cfg) == State.DISARMED
    assert step(State.DISARMED, w, rc_ok(NOW, mode_auto=False, arm=True), NOW, cfg) == State.MANUAL


def test_black_box_can_block_or_revoke_auto(cfg):
    w = world(NOW)
    assert step(State.MANUAL, w, rc_ok(NOW, allowed_auto=False), NOW, cfg) == State.MANUAL
    assert step(State.AUTO, w, rc_ok(NOW, allowed_auto=False), NOW, cfg) == State.URGENT_STOP
