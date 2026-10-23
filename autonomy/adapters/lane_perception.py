"""Lane detection as a perception adapter: camera frames -> LaneDetector -> lane Path.

Wraps lane_detection (imported, not copied). Frames come from any iterable of BGR images,
usually lane_detection.sources.open_source(...): a recorded clip, an image, a webcam, or the OAK-D.
LaneResult.center is already in the car frame (meters, x right, y forward, origin on the
ground below the camera), so it becomes a Path directly via planner.lane_path_from_poly.

An optional obstacle source (another perception adapter) supplies obstacles; their detector
boxes are passed to LaneDetector as exclude_boxes so white cones and karts aren't read as lines.
Known gap: the detector's 16:9 input and a 4:3 lane frame may crop differently, so boxes are
assumed to already be in the lane frame's normalized coordinates.

Timing: with fps set (recorded video), frame i is stamped i / fps and frames are consumed as
simulated time passes, so a clip replays at its real speed under SimClock. With fps=None
(live camera), each tick pulls the next frame and stamps it with the current time. When the
frames run out, the timestamp stops advancing and the supervisor sees stale data.
"""
from pathlib import Path as FsPath
from typing import Iterable, Iterator, Optional

import cv2
import numpy as np

from lane_detection import LaneConfig, LaneDetector, LaneResult
from lane_detection.sources import open_source

from ..config import Config
from ..planner import lane_path_from_poly
from ..types import PerceptionFrame


def load_lane_config(path) -> LaneConfig:
    """LaneConfig.load silently falls back to defaults for a missing file (handy for the
    calibration tools); the driving stack needs a real calibration, so require the file."""
    path = FsPath(path)
    if not path.is_file():
        raise FileNotFoundError(f"Lane config not found: {path}\n"
                                "Run lane_detection.calibrate_ipm and tune_hsv first (see lane_detection/README.md).")
    cfg = LaneConfig.load(path)
    if cfg.src_pts is None:
        raise ValueError(f"{path} has no IPM calibration (src_pts). Run lane_detection.calibrate_ipm.")
    return cfg


def video_fps(source: str, default: float = 30.0) -> Optional[float]:
    """FPS of a video file, None for live sources ('oak' or a webcam index)."""
    if source == "oak" or source.isdigit():
        return None
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    return fps if fps and fps > 0 else default


class LanePerception:
    def __init__(self, cfg: Config, frames: Iterable[np.ndarray], detector: LaneDetector,
                 fps: Optional[float] = None, obstacle_source=None):
        self.cfg = cfg
        self.frames: Iterator[np.ndarray] = iter(frames)
        self.detector = detector
        self.fps = fps
        self.obstacle_source = obstacle_source
        self.frame_index = -1          # index of the newest frame read
        self.last_t = float("-inf")    # timestamp of the newest processed frame
        self.last_result: Optional[LaneResult] = None
        self.last_lane_path = None
        self.exhausted = False

    @classmethod
    def from_source(cls, cfg: Config, source: str, lane_config_path, width: int = 640, height: int = 480,
                    obstacle_source=None) -> "LanePerception":
        detector = LaneDetector(load_lane_config(lane_config_path))
        return cls(cfg, open_source(source, width, height), detector, video_fps(source), obstacle_source)

    def _next_frame(self) -> Optional[np.ndarray]:
        try:
            frame = next(self.frames)
        except StopIteration:
            self.exhausted = True
            return None
        self.frame_index += 1
        return frame

    def _frame_due(self, now: float) -> Optional[np.ndarray]:
        """Newest frame whose timestamp is <= now (video), or simply the next frame (live)."""
        if self.exhausted:
            return None
        if self.fps is None:
            return self._next_frame()
        frame = None
        while not self.exhausted and (self.frame_index + 1) / self.fps <= now:
            nxt = self._next_frame()
            frame = nxt if nxt is not None else frame
        return frame

    def run(self, now: float) -> PerceptionFrame:
        obstacles, t_obs = [], None
        if self.obstacle_source is not None:
            obs_frame = self.obstacle_source.run(now)
            obstacles, t_obs = obs_frame.obstacles, obs_frame.t

        frame = self._frame_due(now)
        if frame is not None:
            boxes = [o.box for o in obstacles if o.box is not None]
            self.last_result = self.detector.process(frame, exclude_boxes=boxes or None)
            self.last_t = now if self.fps is None else self.frame_index / self.fps
            res = self.last_result
            self.last_lane_path = lane_path_from_poly(res.center, res.confidence, self.cfg) if res.ok else None

        t = self.last_t if t_obs is None else min(self.last_t, t_obs)
        return PerceptionFrame(t=t, obstacles=list(obstacles), lane_path=self.last_lane_path)
