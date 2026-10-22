import numpy as np
import pytest

from autonomy.planner import plan

from .conftest import obs


def test_cone_gate_midpoints(cfg):
    cones = [obs("cone", -0.5, 2.0), obs("cone", 0.7, 2.0), obs("cone", -0.4, 4.0), obs("cone", 0.8, 4.0)]
    path = plan(cones, cfg)
    assert path.source == "cones"
    assert path.confidence == pytest.approx(min(1.0, 2 / cfg.planner.full_confidence_pairs))
    np.testing.assert_allclose(path.points, [[0, 0], [0.1, 2.0], [0.2, 4.0]])


def test_only_left_cones_offset_path(cfg):
    half = cfg.planner.cone_track_width_m / 2
    cones = [obs("cone", -0.6, 3.0), obs("cone", -0.5, 1.5)]
    path = plan(cones, cfg)
    assert path.source == "cones_left"
    assert path.confidence == cfg.planner.one_side_confidence
    np.testing.assert_allclose(path.points, [[0, 0], [-0.5 + half, 1.5], [-0.6 + half, 3.0]])


def test_only_right_cones_offset_path(cfg):
    half = cfg.planner.cone_track_width_m / 2
    path = plan([obs("cone", 0.6, 2.0)], cfg)
    assert path.source == "cones_right"
    np.testing.assert_allclose(path.points[-1], [0.6 - half, 2.0])


def test_unpaired_extra_cones_extend_path(cfg):
    half = cfg.planner.cone_track_width_m / 2
    cones = [obs("cone", -0.5, 2.0), obs("cone", 0.5, 2.0), obs("cone", -0.5, 4.0)]
    path = plan(cones, cfg)
    np.testing.assert_allclose(path.points, [[0, 0], [0, 2.0], [-0.5 + half, 4.0]])


def test_no_cones_no_lane_zero_confidence(cfg):
    path = plan([obs("person", 0, 3)], cfg)
    assert path.confidence == 0 and path.source == "none" and len(path.points) == 0


def test_cones_behind_ignored(cfg):
    assert plan([obs("cone", -0.5, -1.0)], cfg).confidence == 0


def test_lane_center_used(cfg):
    lane = np.array([[0.0, 3.0], [0.0, 0.5], [0.1, 1.5]])
    path = plan([], cfg, lane_center=lane, lane_confidence=0.8)
    assert path.source == "lane" and path.confidence == 0.8
    assert list(path.points[:, 1]) == [0.5, 1.5, 3.0]


def test_preference_between_lane_and_cones(cfg):
    lane = np.array([[0.0, 0.5], [0.0, 3.0]])
    cones = [obs("cone", -0.5, 2.0), obs("cone", 0.5, 2.0)]
    assert plan(cones, cfg, lane, 0.8).source == ("lane" if cfg.planner.prefer == "lane" else "cones")
    assert plan(cones, cfg, lane, 0.0).source == "cones"   # unusable lane falls back to cones
