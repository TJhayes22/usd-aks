import numpy as np
import pytest

from autonomy.geometry import (corridor_half_width, corridor_polyline, distance_to_polyline,
                               relevant_obstacles, stop_threshold, stopping_distance)

from .conftest import obs, straight_path


def test_stopping_distance_formula():
    # 2 m/s, 0.19 s latency, 3 m/s^2: 0.38 + 4/6
    assert stopping_distance(2.0, 0.19, 3.0) == pytest.approx(0.38 + 2.0 / 3.0)
    assert stopping_distance(0.0, 0.19, 3.0) == 0.0


def test_stop_threshold_adds_class_margin(cfg):
    base = stopping_distance(1.0, cfg.vehicle.latency_s, cfg.vehicle.brake_decel_mps2)
    assert stop_threshold("person", 1.0, cfg) == pytest.approx(base + cfg.obstacles.stop_margin_m["person"])


def test_distance_to_polyline():
    poly = np.array([[0.0, 0.0], [0.0, 10.0]])
    assert distance_to_polyline((0.5, 3.0), poly) == pytest.approx(0.5)
    assert distance_to_polyline((0.0, 12.0), poly) == pytest.approx(2.0)


def test_empty_path_corridor_is_straight_ahead(cfg):
    poly = corridor_polyline(np.zeros((0, 2)), cfg.obstacles.corridor_length_m)
    assert poly.tolist() == [[0.0, 0.0], [0.0, cfg.obstacles.corridor_length_m]]


def test_corridor_extends_past_short_path(cfg):
    path = straight_path(length=2.0)
    far_person = obs("person", 0.0, 6.0)
    assert relevant_obstacles([far_person], path.points, cfg) == [far_person]


def test_corridor_filter(cfg):
    half = corridor_half_width(cfg)
    inside = obs("person", half * 0.5, 3.0)
    outside = obs("person", half * 2.0, 3.0)
    behind = obs("person", 0.0, -1.0)
    cone = obs("cone", 0.0, 2.0)
    kart = obs("kart", 0.0, 1.0)
    got = relevant_obstacles([inside, outside, behind, cone, kart], straight_path().points, cfg)
    assert got == [kart, inside]   # nearest first, cones excluded


def test_corridor_follows_curved_path(cfg):
    pts = np.array([[0.0, 0.0], [0.5, 1.0], [1.5, 2.0], [3.0, 3.0]])
    on_curve = obs("person", 1.5, 2.0)
    straight_ahead = obs("person", 0.0, 2.0)
    assert relevant_obstacles([on_curve, straight_ahead], pts, cfg) == [on_curve]
