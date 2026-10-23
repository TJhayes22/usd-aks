"""Lane detection as a path source: planner priority, low-confidence SLOW, YOLO box masking.

Camera images come from lane_detection.selftest's synthetic pinhole camera (no hardware).
"""
import numpy as np
import pytest

from autonomy.adapters.lane_perception import LanePerception, load_lane_config
from autonomy.adapters.mock_perception import MockPerception
from autonomy.adapters.mock_rc import MockRc, arm_then_auto
from autonomy.adapters.mock_vehicle import MockVehicle
from autonomy.loop import Loop, SimClock
from autonomy.planner import lane_path_from_poly, plan
from autonomy.supervisor import Supervisor
from autonomy.types import State, World
from lane_detection import LaneDetector
from lane_detection.selftest import H, W, make_cfg, project, render

from .conftest import obs, rc_ok

LANE_HALF = 0.46 / 2       # selftest renders lines at +/- lane_width/2
NOW = 10.0


def straight_frames(n, offset=0.0):
    return [render(lambda y: offset + 0 * y, seed=i) for i in range(n)]


def right_line_box():
    """Normalized image box around the right tape line over the bird's-eye range (0.25-3 m)."""
    xs = np.array([LANE_HALF - 0.05, LANE_HALF + 0.05] * 2)
    ys = np.array([0.25, 0.25, 3.0, 3.0])
    uv = project(xs, ys)
    (u0, v0), (u1, v1) = uv.min(0), uv.max(0)
    return (max(u0 / W, 0.0), max(v0 / H, 0.0), min(u1 / W, 1.0), min(v1 / H, 1.0))


# ---------------------------------------------------------------- polynomial -> Path
def test_lane_poly_sampled_into_path(cfg):
    path = lane_path_from_poly([0.02, 0.05, 0.10], 0.8, cfg)
    assert path.source == "lanes" and path.confidence == 0.8
    ys = path.points[:, 1]
    assert ys[0] == pytest.approx(cfg.planner.lane_sample_start_m)
    assert ys[-1] == pytest.approx(cfg.control.ld_max_m)
    np.testing.assert_allclose(path.points[:, 0], 0.02 * ys**2 + 0.05 * ys + 0.10)


def test_no_lane_or_zero_confidence_gives_empty_path(cfg):
    assert lane_path_from_poly(None, 0.9, cfg).confidence == 0
    assert lane_path_from_poly([0, 0, 0], 0.0, cfg).confidence == 0


# ---------------------------------------------------------------- planner priority
def test_cones_on_both_sides_beat_lanes(cfg):
    cones = [obs("cone", -0.5, 2.0), obs("cone", 0.5, 2.0)]
    lane = lane_path_from_poly([0, 0, 0.3], 1.0, cfg)
    assert plan(cones, cfg, lane).source == "cones"


def test_lanes_only_gives_lane_path(cfg):
    lane = lane_path_from_poly([0, 0, 0.3], 0.9, cfg)
    path = plan([obs("person", 3.0, 4.0)], cfg, lane)
    assert path.source == "lanes"
    np.testing.assert_allclose(path.points[:, 0], 0.3)


def test_lanes_beat_one_sided_cones_and_fall_back_to_them(cfg):
    left_only = [obs("cone", -0.5, 2.0)]
    assert plan(left_only, cfg, lane_path_from_poly([0, 0, 0], 0.9, cfg)).source == "lanes"
    assert plan(left_only, cfg, lane_path_from_poly([0, 0, 0], 0.0, cfg)).source == "cones_left"
    assert plan([], cfg, None).source == "none"


# ---------------------------------------------------------------- supervisor
def test_low_lane_confidence_slows_after_n_ticks(cfg):
    weak = cfg.supervisor.min_path_confidence / 2
    n = cfg.supervisor.low_confidence_ticks
    sup = Supervisor(cfg, initial=State.AUTO)
    period = 1.0 / cfg.loop.hz
    states = []
    for i in range(n + 1):
        t = NOW + i * period
        path = plan([], cfg, lane_path_from_poly([0, 0, 0], weak, cfg))
        states.append(sup.run(World(t_perception=t, path=path), rc_ok(t), t))
    assert states[: n - 1] == [State.AUTO] * (n - 1)
    assert states[n - 1:] == [State.SLOW, State.SLOW]


# ---------------------------------------------------------------- adapter
def test_yolo_box_over_a_line_masks_it(cfg):
    box = right_line_box()
    frames = straight_frames(1)
    clean = LanePerception(cfg, frames, LaneDetector(make_cfg()), fps=10.0)
    clean.run(0.0)
    assert clean.last_result.source == "both"

    cone_on_line = MockPerception([(0.0, [("cone", LANE_HALF, 1.0, box)])])
    masked = LanePerception(cfg, frames, LaneDetector(make_cfg()), fps=10.0, obstacle_source=cone_on_line)
    frame = masked.run(0.0)
    assert masked.last_result.source == "left_only"     # right line ignored
    assert frame.lane_path.source == "lanes"
    assert frame.obstacles[0].box == box               # obstacles still reach the planner/supervisor


def test_video_frames_follow_simulated_time(cfg):
    perc = LanePerception(cfg, straight_frames(3), LaneDetector(make_cfg()), fps=10.0)
    assert perc.run(0.05).t == 0.0             # frame 0 due at t=0
    assert perc.run(0.25).t == pytest.approx(0.2)   # frames 1 and 2 due; newest wins
    perc.run(1.0)                              # clip used up: timestamp stops advancing
    assert perc.exhausted and perc.last_t == pytest.approx(0.2)


def test_lane_config_must_exist_and_be_calibrated(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_lane_config(tmp_path / "missing.json")
    (tmp_path / "blank.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="calibration"):
        load_lane_config(tmp_path / "blank.json")


def test_end_to_end_lane_offset_steers_right(cfg):
    """Recorded-clip style run: lane center 10 cm right -> AUTO, steering right."""
    clock = SimClock()
    perc = LanePerception(cfg, straight_frames(10, offset=0.10), LaneDetector(make_cfg()), fps=10.0)
    loop = Loop(cfg, perc, MockRc(arm_then_auto()), MockVehicle(cfg), clock=clock, sleep=clock.sleep)
    history = loop.run(duration_s=0.9)
    driving = [h for h in history if h.state == State.AUTO]
    assert driving, [h.state for h in history]
    assert all(h.path_confidence > 0 for h in driving)
    assert driving[-1].cmd.steer_rad > 0
