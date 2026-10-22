import logging

import pytest

from autonomy.adapters.csv_replay import CsvReplayPerception
from autonomy.adapters.mock_perception import MockPerception, cone_gates
from autonomy.adapters.mock_rc import MockRc, arm_then_auto
from autonomy.adapters.mock_vehicle import MockVehicle
from autonomy.loop import Loop, SimClock
from autonomy.types import Command, State


def make_loop(cfg, perception, rc_script):
    clock = SimClock()
    vehicle = MockVehicle(cfg)
    loop = Loop(cfg, perception, MockRc(rc_script), vehicle, clock=clock, sleep=clock.sleep)
    return loop, vehicle


def states(history):
    out = []
    for h in history:
        if not out or out[-1] != h.state:
            out.append(h.state)
    return out


def test_end_to_end_person_appears(cfg):
    perception = MockPerception([(0.0, cone_gates()), (3.0, cone_gates() + [("person", 0.0, 2.5)])])
    loop, vehicle = make_loop(cfg, perception, arm_then_auto())
    history = loop.run(duration_s=4.0)
    seq = states(history)
    assert seq[:3] == [State.MANUAL, State.AUTO, State.URGENT_STOP] or \
        seq[:4] == [State.MANUAL, State.AUTO, State.SLOW, State.URGENT_STOP]
    assert max(h.speed_mps for h in history) > 0.5
    assert history[-1].cmd == Command(0.0, 0.0, 1.0)
    assert vehicle.speed_mps < max(h.speed_mps for h in history)


def test_camera_dropout_faults_then_stops(cfg):
    perception = MockPerception([(0.0, cone_gates()), (1.0, None)])
    loop, _ = make_loop(cfg, perception, arm_then_auto())
    seq = states(loop.run(duration_s=2.5))
    assert seq[-2:] == [State.FAULT, State.URGENT_STOP]


def test_rc_loss_stops(cfg):
    perception = MockPerception([(0.0, cone_gates())])
    loop, _ = make_loop(cfg, perception, arm_then_auto() + [(1.0, {"link_ok": False})])
    assert states(loop.run(duration_s=2.0))[-1] == State.URGENT_STOP


def test_overrun_warns(cfg, caplog):
    clock = SimClock()
    period = 1.0 / cfg.loop.hz

    class SlowPerception(MockPerception):
        def run(self, now):
            clock.sleep(1.5 * period)
            return super().run(now)

    loop = Loop(cfg, SlowPerception([(0.0, [])]), MockRc(arm_then_auto()), MockVehicle(cfg),
                clock=clock, sleep=clock.sleep)
    with caplog.at_level(logging.WARNING, logger="autonomy.loop"):
        loop.run(n_ticks=3)
    assert loop.overruns == 3
    assert "overrun" in caplog.text


def test_exception_brakes_vehicle(cfg):
    class Broken(MockPerception):
        def run(self, now):
            raise RuntimeError("camera exploded")

    clock = SimClock()
    vehicle = MockVehicle(cfg)
    loop = Loop(cfg, Broken([]), MockRc(arm_then_auto()), vehicle, clock=clock, sleep=clock.sleep)
    with pytest.raises(RuntimeError):
        loop.run(n_ticks=1)
    assert vehicle.log[-1][1] == Command(0.0, 0.0, 1.0)


def test_csv_replay_parsing(cfg, tmp_path):
    p = tmp_path / "obstacles.csv"
    p.write_text("ts,label,conf,forward_m,lateral_m\n"
                 "1000.0,person,0.7,0.85,-0.55\n"
                 "1000.0,cone,0.6,5.0,0.9\n"
                 "1000.2,kart,0.8,3.0,0.1\n")
    perc = CsvReplayPerception(p, cfg)
    assert perc.run(-0.1).t == float("-inf")
    f0 = perc.run(0.1)
    assert f0.t == 0.0 and [o.cls for o in f0.obstacles] == ["person", "cone"]
    assert f0.obstacles[0].x == pytest.approx(cfg.replay.csv_lateral_sign * -0.55)
    assert f0.obstacles[0].y == pytest.approx(0.85)
    f1 = perc.run(5.0)
    assert f1.t == pytest.approx(0.2) and f1.obstacles[0].cls == "kart"
