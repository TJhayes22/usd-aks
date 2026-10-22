"""
Classical track-line detection for the 1/10 AKS kart.

Pipeline:
    (undistort) -> HSV threshold -> bird's-eye warp (IPM)
    -> search around last frame's fit, or sliding windows if lost
    -> quadratic fit per line -> lane center + control metrics

Output frame (meters, on the ground plane):
    origin = point on the ground directly below the camera
    y = forward, x = right
    every fit is x = a*y^2 + b*y + c
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

SIDES = ("left", "right")


# --------------------------------------------------------------------------- config
@dataclass
class LaneConfig:
    # Color threshold, OpenCV HSV ranges (H 0-179, S/V 0-255). Defaults ~ white tape.
    # If hsv_lower[0] > hsv_upper[0], hue wraps around 0 (use for red tape).
    hsv_lower: list = field(default_factory=lambda: [0, 0, 180])
    hsv_upper: list = field(default_factory=lambda: [179, 60, 255])

    # IPM calibration (written by calibrate_ipm.py)
    image_size: list | None = None        # [w, h] of the frames used to calibrate
    src_pts: list | None = None           # image px: near-L, near-R, far-R, far-L
    rect_width_m: float = 0.60
    rect_length_m: float = 0.90
    rect_near_dist_m: float = 0.30        # camera (on ground) -> near edge of rectangle
    rect_lateral_offset_m: float = 0.0    # rectangle center, + = right of camera
    intrinsics_path: str | None = None    # JSON with K, D, size -> enables undistortion

    # Bird's-eye canvas
    bev_width_m: float = 2.0
    bev_length_m: float = 3.0
    px_per_m: float = 200.0

    # Lane model
    lane_width_m: float = 0.46            # 1/10 of AKS 15 ft minimum. Measure your track!
    n_windows: int = 12
    window_margin_m: float = 0.08
    min_window_px: int = 30
    min_line_px: int = 200
    search_margin_m: float = 0.08
    lookahead_m: float = 1.0
    smoothing: float = 0.6                # EMA weight on previous fit (0 = off)
    max_misses: int = 5                   # frames a line can vanish before its prior is dropped

    @classmethod
    def load(cls, path: str | Path) -> "LaneConfig":
        """Missing file -> defaults (calibrate_ipm/tune_hsv start from scratch this way).
        A relative intrinsics_path is resolved against the config file's own folder, so the
        result doesn't depend on the launch directory."""
        cfg = cls()
        p = Path(path)
        if p.exists():
            for k, v in json.loads(p.read_text(encoding="utf-8")).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            if cfg.intrinsics_path:
                rel = Path(str(cfg.intrinsics_path).replace("\\", "/"))   # tolerate Windows-written paths
                cfg.intrinsics_path = str((p.resolve().parent / rel).resolve())
        return cfg

    def save(self, path: str | Path) -> None:
        """intrinsics_path is written relative to the config file's folder, with forward slashes."""
        p = Path(path)
        data = asdict(self)
        if self.intrinsics_path:
            target = Path(self.intrinsics_path).resolve()
            try:
                data["intrinsics_path"] = Path(os.path.relpath(target, p.resolve().parent)).as_posix()
            except ValueError:   # different drive on Windows: keep it absolute
                data["intrinsics_path"] = target.as_posix()
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- helpers
def threshold_hsv(frame_bgr: np.ndarray, lower, upper) -> np.ndarray:
    """Binary mask (0/255) of pixels inside the HSV range. Handles hue wraparound."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lo = np.array(lower, np.uint8)
    hi = np.array(upper, np.uint8)
    if lo[0] <= hi[0]:
        return cv2.inRange(hsv, lo, hi)
    a = cv2.inRange(hsv, lo, np.array([179, hi[1], hi[2]], np.uint8))
    b = cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8), hi)
    return a | b


def bev_size(cfg: LaneConfig) -> tuple[int, int]:
    return int(round(cfg.bev_width_m * cfg.px_per_m)), int(round(cfg.bev_length_m * cfg.px_per_m))


def ground_to_bev_px(cfg: LaneConfig, x_m, y_m):
    w, h = bev_size(cfg)
    return w / 2 + np.asarray(x_m) * cfg.px_per_m, h - np.asarray(y_m) * cfg.px_per_m


def compute_homography(cfg: LaneConfig, frame_size: tuple[int, int]) -> np.ndarray:
    """Image px -> BEV px homography from the calibration rectangle."""
    if cfg.src_pts is None:
        raise ValueError("No IPM calibration in config. Run calibrate_ipm.py first.")
    src = np.float32(cfg.src_pts)
    if cfg.image_size and tuple(cfg.image_size) != tuple(frame_size):
        src *= np.float32([frame_size[0] / cfg.image_size[0], frame_size[1] / cfg.image_size[1]])
    cx, hw = cfg.rect_lateral_offset_m, cfg.rect_width_m / 2
    near, far = cfg.rect_near_dist_m, cfg.rect_near_dist_m + cfg.rect_length_m
    xs = [cx - hw, cx + hw, cx + hw, cx - hw]
    ys = [near, near, far, far]
    u, v = ground_to_bev_px(cfg, xs, ys)
    dst = np.float32(np.stack([u, v], 1))
    return cv2.getPerspectiveTransform(src, dst)


class Undistorter:
    """Optional lens undistortion (matters a lot for the OAK-D *Wide* lens)."""

    def __init__(self, intrinsics_path: str | None):
        self.K = self.D = None
        self._maps = None
        self._size = None
        if intrinsics_path:
            data = json.loads(Path(intrinsics_path).read_text(encoding="utf-8"))
            self.K = np.array(data["K"], np.float64)
            self.D = np.array(data["D"], np.float64)
            self.K_size = tuple(data["size"])

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if self.K is None:
            return frame
        h, w = frame.shape[:2]
        if self._size != (w, h):
            K = self.K.copy()
            K[0] *= w / self.K_size[0]
            K[1] *= h / self.K_size[1]
            self._maps = cv2.initUndistortRectifyMap(K, self.D, None, K, (w, h), cv2.CV_16SC2)
            self._size = (w, h)
        return cv2.remap(frame, *self._maps, cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- result
@dataclass
class LaneResult:
    left: np.ndarray | None = None
    right: np.ndarray | None = None
    center: np.ndarray | None = None
    lateral_offset_m: float | None = None   # + = lane center is RIGHT of camera (steer right)
    heading_error_rad: float | None = None  # + = lane heads to the right of camera forward
    curvature_1pm: float | None = None      # at lookahead, + = bending right
    target_m: tuple | None = None           # (x, y) on center line at lookahead -> pure pursuit
    width_m: float | None = None            # measured lane width (both lines only)
    confidence: float = 0.0
    source: str = "none"                    # "both", "left_only", "right_only", "none"

    @property
    def ok(self) -> bool:
        return self.center is not None


# --------------------------------------------------------------------------- detector
class LaneDetector:
    def __init__(self, cfg: LaneConfig):
        self.cfg = cfg
        self.ppm = cfg.px_per_m
        self.bev_w, self.bev_h = bev_size(cfg)
        self.undistort = Undistorter(cfg.intrinsics_path)
        self._frame_size = None
        self.M = self.Minv = None
        self._prior = {s: None for s in SIDES}
        self._misses = {s: 0 for s in SIDES}
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.last_frame = self.last_mask = self.last_bev = None

    def reset(self):
        self._prior = {s: None for s in SIDES}
        self._misses = {s: 0 for s in SIDES}

    # ---- stages
    def threshold(self, frame: np.ndarray) -> np.ndarray:
        mask = threshold_hsv(frame, self.cfg.hsv_lower, self.cfg.hsv_upper)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

    def warp(self, img: np.ndarray, interp=cv2.INTER_NEAREST) -> np.ndarray:
        return cv2.warpPerspective(img, self.M, (self.bev_w, self.bev_h), flags=interp)

    def process(self, frame: np.ndarray, exclude_boxes=None) -> LaneResult:
        """exclude_boxes: optional list of (xmin, ymin, xmax, ymax), normalized 0-1 (e.g. YOLO
        detections). Those regions are blanked from the mask so cones/karts aren't read as lines."""
        cfg = self.cfg
        h, w = frame.shape[:2]
        if self._frame_size != (w, h):
            self.M = compute_homography(cfg, (w, h))
            self.Minv = np.linalg.inv(self.M)
            self._frame_size = (w, h)

        frame = self.undistort(frame)
        mask = self.threshold(frame)
        if exclude_boxes:
            mh, mw = mask.shape
            for x0, y0, x1, y1 in exclude_boxes:
                pad = 0.1 * max(x1 - x0, y1 - y0)
                mask[max(0, int((y0 - pad) * mh)):min(mh, int((y1 + pad) * mh)),
                     max(0, int((x0 - pad) * mw)):min(mw, int((x1 + pad) * mw))] = 0
        bev = self.warp(mask)
        self.last_frame, self.last_mask, self.last_bev = frame, mask, bev

        ys, xs = bev.nonzero()
        x_m = (xs - self.bev_w / 2) / self.ppm
        y_m = (self.bev_h - ys) / self.ppm

        # 1) cheap path: search around last frame's fits
        sel = {}
        for s in SIDES:
            if self._prior[s] is not None:
                m = np.abs(x_m - np.polyval(self._prior[s], y_m)) < cfg.search_margin_m
                if m.sum() >= cfg.min_line_px:
                    sel[s] = m

        # 2) recovery path: sliding windows for any side still missing
        if len(sel) < 2:
            sel.update(self._sliding_window_search(x_m, y_m, sel))

        # 3) fit + sanity check
        fits, counts = {}, {}
        for s, m in sel.items():
            f = self._fit(x_m[m], y_m[m])
            if f is not None:
                fits[s], counts[s] = f, int(m.sum())
        fits = self._sanity(fits, counts)

        # 4) temporal smoothing
        a = cfg.smoothing
        for s in SIDES:
            if s in fits:
                prev = self._prior[s]
                fits[s] = fits[s] if prev is None else a * prev + (1 - a) * fits[s]
                self._prior[s] = fits[s]
                self._misses[s] = 0
            else:
                self._misses[s] += 1
                if self._misses[s] > cfg.max_misses:
                    self._prior[s] = None

        return self._result(fits, counts)

    # ---- internals
    def _sliding_window_search(self, x_m, y_m, already: dict) -> dict:
        cfg = self.cfg
        avail = np.ones(len(x_m), bool)
        for m in already.values():
            avail &= ~m

        # histogram of the near half to find line bases
        near = avail & (y_m < cfg.bev_length_m / 2)
        bins = np.arange(-cfg.bev_width_m / 2, cfg.bev_width_m / 2 + 1e-9, 0.02)
        hist, edges = np.histogram(x_m[near], bins=bins)
        hist = np.convolve(hist, np.ones(3), mode="same")
        centers = (edges[:-1] + edges[1:]) / 2
        bases = []
        for _ in range(2):
            i = int(np.argmax(hist))
            if hist[i] < cfg.min_window_px:
                break
            bases.append(float(centers[i]))
            hist[np.abs(centers - centers[i]) < 0.6 * cfg.lane_width_m] = 0

        cands = [(b, self._track_windows(x_m, y_m, avail, b)) for b in bases]
        cands = [(b, m) for b, m in cands if m.sum() >= cfg.min_line_px]
        if not cands:
            return {}

        # assign candidates to sides using priors (or the other line) as reference
        lw = cfg.lane_width_m
        y_ref = cfg.rect_near_dist_m + 0.2
        refs = {s: float(np.polyval(self._prior[s], y_ref)) for s in SIDES if self._prior[s] is not None}
        if "left" not in refs and "right" in refs:
            refs["left"] = refs["right"] - lw
        if "right" not in refs and "left" in refs:
            refs["right"] = refs["left"] + lw
        if not refs:
            refs = {"left": -lw / 2, "right": lw / 2}

        missing = [s for s in SIDES if s not in already]
        out = {}
        if len(cands) == 2 and len(missing) == 2:
            cands.sort(key=lambda c: c[0])
            return {"left": cands[0][1], "right": cands[1][1]}
        for b, m in sorted(cands, key=lambda c: c[1].sum(), reverse=True):
            free = [s for s in missing if s not in out]
            if not free:
                break
            s = min(free, key=lambda s: abs(b - refs[s]))
            out[s] = m
        return out

    def _track_windows(self, x_m, y_m, avail, base) -> np.ndarray:
        cfg = self.cfg
        win_h = cfg.bev_length_m / cfg.n_windows
        x_cur, dx = base, 0.0
        sel = np.zeros(len(x_m), bool)
        for i in range(cfg.n_windows):
            in_win = avail & (y_m >= i * win_h) & (y_m < (i + 1) * win_h) & (np.abs(x_m - x_cur) < cfg.window_margin_m)
            if in_win.sum() >= cfg.min_window_px:
                sel |= in_win
                new = float(x_m[in_win].mean())
                dx = new - x_cur
                x_cur = new
            else:
                x_cur += dx  # keep following the curve through gaps
        return sel

    def _fit(self, x, y):
        if len(x) < self.cfg.min_line_px:
            return None
        span = y.max() - y.min()
        if span < 0.15:
            return None
        if span > 0.6:
            return np.polyfit(y, x, 2)
        return np.array([0.0, *np.polyfit(y, x, 1)])  # short segment: straight line only

    def _sanity(self, fits: dict, counts: dict) -> dict:
        if "left" in fits and "right" in fits:
            lw = self.cfg.lane_width_m
            y_ref = self.cfg.rect_near_dist_m + 0.2
            width = np.polyval(fits["right"], y_ref) - np.polyval(fits["left"], y_ref)
            if not (0.5 * lw < width < 1.6 * lw):
                drop = "left" if counts["left"] < counts["right"] else "right"
                fits.pop(drop)
        return fits

    def _result(self, fits: dict, counts: dict) -> LaneResult:
        cfg = self.cfg
        L, R = fits.get("left"), fits.get("right")
        half = np.array([0.0, 0.0, cfg.lane_width_m / 2])
        res = LaneResult(left=L, right=R)
        conf = {s: min(1.0, counts[s] / (4 * cfg.min_line_px)) for s in fits}

        if L is not None and R is not None:
            C = (L + R) / 2
            res.width_m = float(np.polyval(R, cfg.rect_near_dist_m + 0.2) - np.polyval(L, cfg.rect_near_dist_m + 0.2))
            res.confidence, res.source = (conf["left"] + conf["right"]) / 2, "both"
        elif L is not None:
            C = L + half
            res.confidence, res.source = 0.5 * conf["left"], "left_only"
        elif R is not None:
            C = R - half
            res.confidence, res.source = 0.5 * conf["right"], "right_only"
        else:
            return res

        a, b, c = C
        la = cfg.lookahead_m
        slope = 2 * a * la + b
        res.center = C
        res.lateral_offset_m = float(c)
        res.heading_error_rad = float(np.arctan(b))
        res.curvature_1pm = float(2 * a / (1 + slope**2) ** 1.5)
        res.target_m = (float(np.polyval(C, la)), float(la))
        return res

    # ---- debug drawing
    def draw(self, res: LaneResult) -> tuple[np.ndarray, np.ndarray]:
        """Returns (camera overlay, bird's-eye debug view)."""
        cfg = self.cfg
        overlay = self.last_frame.copy()
        overlay[self.last_mask > 0] = (0, 0, 255)
        overlay = cv2.addWeighted(self.last_frame, 0.6, overlay, 0.4, 0)
        bev = cv2.cvtColor(self.last_bev, cv2.COLOR_GRAY2BGR)

        ys = np.linspace(cfg.rect_near_dist_m * 0.5, cfg.bev_length_m, 60)
        colors = {"left": (255, 140, 0), "right": (0, 140, 255), "center": (0, 255, 0)}
        for name, fit in (("left", res.left), ("right", res.right), ("center", res.center)):
            if fit is None:
                continue
            u, v = ground_to_bev_px(cfg, np.polyval(fit, ys), ys)
            pts = np.stack([u, v], 1).astype(np.float32)
            cv2.polylines(bev, [pts.astype(np.int32)], False, colors[name], 2)
            img = cv2.perspectiveTransform(pts[None], self.Minv)[0]
            img = img[np.all(np.isfinite(img), 1) & np.all(np.abs(img) < 1e4, 1)]
            if len(img) > 1:
                cv2.polylines(overlay, [img.astype(np.int32)], False, colors[name], 3)

        if res.target_m is not None:
            u, v = ground_to_bev_px(cfg, *res.target_m)
            cv2.circle(bev, (int(u), int(v)), 6, (0, 255, 255), -1)
        cv2.circle(bev, (self.bev_w // 2, self.bev_h - 3), 6, (255, 255, 255), -1)  # camera

        if res.ok:
            txt = (f"{res.source}  off {res.lateral_offset_m:+.3f}m  "
                   f"hdg {np.degrees(res.heading_error_rad):+.1f}deg  conf {res.confidence:.2f}")
        else:
            txt = "NO LANE"
        cv2.putText(overlay, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(overlay, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return overlay, bev
