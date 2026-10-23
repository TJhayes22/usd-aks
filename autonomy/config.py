"""Load config.yaml into frozen dataclasses. Missing or unknown keys are errors, so typos fail loudly."""
from dataclasses import dataclass, fields
from pathlib import Path as FsPath
from typing import Dict, Optional, Union

import yaml

DEFAULT_PATH = FsPath(__file__).resolve().parent / "config.yaml"


@dataclass(frozen=True)
class VehicleCfg:
    wheelbase_m: float
    track_width_m: float
    vehicle_width_m: float
    camera_to_rear_axle_m: float
    max_steer_rad: float
    max_speed_mps: float
    accel_max_mps2: float
    brake_decel_mps2: float
    latency_s: float


@dataclass(frozen=True)
class LoopCfg:
    hz: float
    history_max_ticks: int


@dataclass(frozen=True)
class TimeoutsCfg:
    rc_timeout_s: float
    perception_timeout_s: float
    fault_escalate_s: float


@dataclass(frozen=True)
class PlannerCfg:
    cone_track_width_m: float
    one_side_confidence: float
    full_confidence_pairs: int
    lane_sample_start_m: float
    lane_sample_step_m: float


@dataclass(frozen=True)
class SupervisorCfg:
    min_path_confidence: float
    low_confidence_ticks: int
    slow_factor: float


@dataclass(frozen=True)
class ObstaclesCfg:
    corridor_margin_m: float
    corridor_length_m: float
    stop_margin_m: Dict[str, float]
    follow_distance_m: float
    k_follow_per_s: float

    def margin(self, cls: str) -> float:
        return self.stop_margin_m.get(cls, self.stop_margin_m["default"])


@dataclass(frozen=True)
class ControlCfg:
    cruise_speed_mps: float
    slow_speed_mps: float
    ld_min_m: float
    ld_max_m: float
    k_ld_s: float
    kp_speed: float
    speed_tolerance_mps: float
    max_throttle: float


@dataclass(frozen=True)
class ReplayCfg:
    csv_lateral_sign: float


@dataclass(frozen=True)
class Config:
    vehicle: VehicleCfg
    loop: LoopCfg
    timeouts: TimeoutsCfg
    planner: PlannerCfg
    supervisor: SupervisorCfg
    obstacles: ObstaclesCfg
    control: ControlCfg
    replay: ReplayCfg


def _build(cls, data: dict, where: str):
    names = {f.name for f in fields(cls)}
    missing = names - set(data)
    unknown = set(data) - names
    if missing or unknown:
        raise ValueError(f"{where}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    return cls(**data)


def from_dict(raw: dict) -> Config:
    groups = {}
    for f in fields(Config):
        if f.name not in raw:
            raise ValueError(f"config: missing section '{f.name}'")
        groups[f.name] = _build(f.type, raw[f.name], f.name)
    unknown = set(raw) - set(groups)
    if unknown:
        raise ValueError(f"config: unknown sections {sorted(unknown)}")
    if "default" not in groups["obstacles"].stop_margin_m:
        raise ValueError("obstacles.stop_margin_m needs a 'default' entry")
    if groups["supervisor"].low_confidence_ticks < 1:
        raise ValueError("supervisor.low_confidence_ticks must be at least 1")
    return Config(**groups)


def load_config(path: Optional[Union[str, FsPath]] = None) -> Config:
    with open(path or DEFAULT_PATH, encoding="utf-8") as f:
        return from_dict(yaml.safe_load(f))
