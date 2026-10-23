"""Disk budget: hitting the cap stops recording only. Driving continues; the warning reaches the
logs and the supervisor's health status; nothing is deleted. Plus deployment-file checks."""
import dataclasses
import logging
import re
import shutil
from collections import namedtuple

import pytest

from autonomy.adapters.mock_perception import MockPerception, cone_gates
from autonomy.adapters.mock_rc import MockRc, arm_then_auto
from autonomy.adapters.mock_vehicle import MockVehicle
from autonomy.adapters.tick_recorder import TickRecorder
from autonomy.loop import Loop, SimClock
from autonomy.paths import REPO
from autonomy.storage import GB, DataBudget, dir_size
from autonomy.types import State

KB = 1024
DiskUsage = namedtuple("DiskUsage", "total used free")


def scenario():
    return [(0.0, cone_gates()), (3.0, cone_gates() + [("person", 0.0, 2.5)])]


def run_loop(cfg, recorder=None, duration=4.0):
    clock = SimClock()
    loop = Loop(cfg, MockPerception(scenario()), MockRc(arm_then_auto()), MockVehicle(cfg),
                recorder=recorder, clock=clock, sleep=clock.sleep)
    return loop, loop.run(duration_s=duration)


# ---------------------------------------------------------------- DataBudget
def test_budget_ok_then_over_cap(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 10 * KB)
    budget = DataBudget(tmp_path, max_bytes=20 * KB, min_free_bytes=0)
    assert budget.check().ok and budget.check().used_bytes == 10 * KB
    budget.add(15 * KB)                      # counted without re-walking the folder
    status = budget.check()
    assert not status.ok and "cap" in status.reason


def test_budget_low_free_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda p: DiskUsage(100 * GB, 99 * GB, 1 * GB))
    status = DataBudget(tmp_path, max_bytes=50 * GB, min_free_bytes=2 * GB).check()
    assert not status.ok and "free" in status.reason


def test_budget_limits_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AKS_DATA_MAX_GB", "0.5")
    monkeypatch.setenv("AKS_DATA_MIN_FREE_GB", "0.25")
    budget = DataBudget(tmp_path)
    assert budget.max_bytes == int(0.5 * GB) and budget.min_free_bytes == int(0.25 * GB)


def test_budget_rescans_on_interval(tmp_path):
    clock = SimClock()
    budget = DataBudget(tmp_path, max_bytes=GB, min_free_bytes=0, clock=clock, rescan_s=10.0)
    assert budget.check().used_bytes == 0
    (tmp_path / "late.bin").write_bytes(b"x" * KB)    # written by someone else
    assert budget.check().used_bytes == 0             # not walked again yet
    clock.sleep(10.0)
    assert budget.check().used_bytes == KB


# ---------------------------------------------------------------- recorder + loop
def test_cap_stops_recording_but_not_driving(cfg, tmp_path, caplog):
    keep = tmp_path / "old_session.bin"
    keep.write_bytes(b"x" * 2 * KB)
    budget = DataBudget(tmp_path, max_bytes=4 * KB, min_free_bytes=0)   # ~60 B/row: trips mid-run
    recorder = TickRecorder(tmp_path / "logs" / "ticks.csv", budget)

    _, baseline = run_loop(cfg)
    with caplog.at_level(logging.WARNING):
        loop, capped = run_loop(cfg, recorder)

    # driving is identical with and without the cap
    assert [(h.state, h.cmd) for h in capped] == [(h.state, h.cmd) for h in baseline]
    assert capped[-1].state == State.URGENT_STOP            # the person still stops the car

    # recording stopped part way, file kept, nothing deleted
    assert 0 < recorder.rows < len(capped)
    assert recorder.stopped_reason and "cap" in recorder.stopped_reason
    assert keep.exists() and (tmp_path / "logs" / "ticks.csv").exists()
    size = (tmp_path / "logs" / "ticks.csv").stat().st_size
    recorder.run(capped[-1])
    assert (tmp_path / "logs" / "ticks.csv").stat().st_size == size

    # the warning is in the supervisor's health status, the tick log and the logs (once)
    health = loop.supervisor.health
    assert not health.ok and any("recording stopped" in w for w in health.warnings)
    assert any(h.warnings for h in capped)
    assert sum("recording stopped" in r.getMessage() for r in caplog.records if r.name == "autonomy.loop") == 1


def test_health_warning_alone_does_not_change_state(cfg):
    class Noisy(MockVehicle):
        warnings = ("recording stopped: test",)

    clock = SimClock()
    loop = Loop(cfg, MockPerception([(0.0, cone_gates())]), MockRc(arm_then_auto()), Noisy(cfg),
                clock=clock, sleep=clock.sleep)
    history = loop.run(duration_s=1.0)
    assert history[-1].state == State.AUTO
    assert loop.supervisor.health.state == State.AUTO and not loop.supervisor.health.ok


def test_tick_csv_contents(cfg, tmp_path):
    recorder = TickRecorder(tmp_path / "ticks.csv", DataBudget(tmp_path, max_bytes=GB, min_free_bytes=0))
    _, history = run_loop(cfg, recorder, duration=0.5)
    recorder.close()
    lines = (tmp_path / "ticks.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("t,state,steer_rad") and len(lines) == len(history) + 1


def test_history_is_bounded(cfg):
    small = dataclasses.replace(cfg, loop=dataclasses.replace(cfg.loop, history_max_ticks=10))
    loop, history = run_loop(small, duration=2.0)
    assert len(history) == 10 and history[-1].t == pytest.approx(loop.clock() - 1 / cfg.loop.hz)


def test_ctrl_c_and_sigterm_brake(cfg):
    for exc in (KeyboardInterrupt, SystemExit):
        class Interrupt(MockPerception):
            def run(self, now, exc=exc):
                if now >= 1.0:
                    raise exc()
                return super().run(now)

        clock = SimClock()
        vehicle = MockVehicle(cfg)
        loop = Loop(cfg, Interrupt([(0.0, cone_gates())]), MockRc(arm_then_auto()), vehicle,
                    clock=clock, sleep=clock.sleep)
        with pytest.raises(exc):
            loop.run(duration_s=2.0)
        assert vehicle.log[-1][1].brake == 1.0 and vehicle.log[-1][1].throttle == 0.0


def test_dir_size(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.bin").write_bytes(b"x" * 300)
    (tmp_path / "y.bin").write_bytes(b"x" * 200)
    assert dir_size(tmp_path) == 500


# ---------------------------------------------------------------- deployment files
def test_deploy_files_have_unix_line_endings():
    for rel in ("scripts/setup_jetson.sh", "deploy/aks.service"):
        assert b"\r\n" not in (REPO / rel).read_bytes(), rel


def test_service_sets_workdir_env_and_entrypoint():
    text = (REPO / "deploy" / "aks.service").read_text(encoding="utf-8")
    for needle in ("WorkingDirectory=%h/AKS_Root", "Environment=AKS_MODELS=", "Environment=AKS_DATA=",
                   "ExecStart=%h/venvs/aks/bin/python -m autonomy.run", "Restart=on-failure"):
        assert needle in text, needle


def test_jetson_requirements_match_laptop_depthai():
    raw = (REPO / "requirements.txt").read_bytes()
    laptop = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8")
    jetson = (REPO / "requirements-jetson.txt").read_text(encoding="utf-8")
    pin = re.compile(r"^depthai==([\w.]+)", re.M)
    assert pin.search(laptop).group(1) == pin.search(jetson).group(1)
    assert "numpy==2.2" in jetson          # numpy 2.3+ has no Python 3.10 wheels
