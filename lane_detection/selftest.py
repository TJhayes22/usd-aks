"""
Synthetic sanity test, no hardware needed: python selftest.py
Simulates a pinhole camera on a 1/10 car looking at two curved tape lines with a
known lateral offset, then checks the detector recovers it.
"""
import numpy as np
import cv2

try:
    from .lane_detector import LaneConfig, LaneDetector
except ImportError:  # run as a plain script
    from lane_detector import LaneConfig, LaneDetector

W, H = 640, 480
F, CAM_H, PITCH = 450.0, 0.18, np.radians(18)  # focal px, camera height m, downward tilt


def project(x, y):
    """Ground point (x right, y forward, meters) -> image px."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    zc = y * np.cos(PITCH) + CAM_H * np.sin(PITCH)     # depth along optical axis
    yc = -y * np.sin(PITCH) + CAM_H * np.cos(PITCH)    # down in camera frame
    return np.stack([W / 2 + F * x / zc, H / 2 + F * yc / zc], -1)


def render(center_fn, lane_w=0.46, tape=0.05, noise=8, seed=0, sides=(-1, 1)):
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), (70, 70, 70), np.uint8)               # sky/background
    ground = project([-3, 3, 3, -3], [0.25, 0.25, 8, 8])
    cv2.fillPoly(img, [ground.astype(np.int32)], (95, 90, 85))   # asphalt
    ys = np.linspace(0.25, 6, 200)
    for side in sides:
        cx = center_fn(ys) + side * lane_w / 2
        poly = np.vstack([project(cx - tape / 2, ys), project(cx + tape / 2, ys)[::-1]])
        cv2.fillPoly(img, [poly.astype(np.int32)], (235, 235, 235))
    img = np.clip(img.astype(int) + rng.integers(-noise, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def make_cfg():
    cfg = LaneConfig()
    cfg.rect_width_m, cfg.rect_length_m, cfg.rect_near_dist_m = 0.6, 0.9, 0.3
    near, far, hw = 0.3, 1.2, 0.3
    cfg.src_pts = project([-hw, hw, hw, -hw], [near, near, far, far]).tolist()
    cfg.image_size = [W, H]
    return cfg


def run_case(name, center_fn, true_off, true_hdg, frames=10, one_side=False):
    det = LaneDetector(make_cfg())
    for i in range(frames):
        img = render(center_fn, seed=i, sides=(-1,) if one_side else (-1, 1))
        res = det.process(img)
    assert res.ok, f"{name}: no lane found"
    err_off = abs(res.lateral_offset_m - true_off)
    err_hdg = abs(np.degrees(res.heading_error_rad - true_hdg))
    status = "PASS" if err_off < 0.03 and err_hdg < 3 else "FAIL"
    print(f"{status} {name:28s} off {res.lateral_offset_m:+.3f} (true {true_off:+.3f})  "
          f"hdg {np.degrees(res.heading_error_rad):+.1f} (true {np.degrees(true_hdg):+.1f})  "
          f"src {res.source}  conf {res.confidence:.2f}")
    overlay, bev = det.draw(res)
    cv2.imwrite(f"selftest_{name.replace(' ', '_')}.png",
                np.hstack([overlay, cv2.resize(bev, (int(bev.shape[1] * H / bev.shape[0]), H))]))
    return status == "PASS"


if __name__ == "__main__":
    ok = all([
        run_case("straight centered", lambda y: 0 * y, 0.0, 0.0),
        run_case("offset right 10cm", lambda y: 0.10 + 0 * y, 0.10, 0.0),
        run_case("curve right + offset", lambda y: 0.08 + 0.05 * y + 0.06 * y**2, 0.08, np.arctan(0.05)),
        run_case("curve left", lambda y: -0.05 - 0.08 * y**2, -0.05, 0.0),
        run_case("left line only", lambda y: 0.05 + 0 * y, 0.05, 0.0, one_side=True),
    ])
    print("\nALL PASS" if ok else "\nSOME FAILED")
