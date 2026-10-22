# lane_detection

Classical track-line detection for the 1/10 AKS kart (OAK-D -> Jetson).
HSV threshold -> bird's-eye warp (IPM) -> sliding windows / search-around-prior -> quadratic fit.
~4 ms/frame at 640x480 on a laptop CPU; no training data needed.

## Files
| File | What it does |
|---|---|
| `lane_detector.py` | Core module: `LaneConfig`, `LaneDetector`, `LaneResult` |
| `sources.py` | Frame sources: OAK-D (DepthAI v3), video, image, webcam; dumps OAK intrinsics |
| `calibrate_ipm.py` | Click a known rectangle on the ground -> bird's-eye calibration |
| `tune_hsv.py` | Trackbars to dial in the tape color |
| `run_lanes.py` | Live/recorded run with debug view, metrics printout, optional recording |
| `selftest.py` | Synthetic camera test, no hardware needed |

## Install
Drop this folder into the repo root (`AKS_Root/lane_detection/`), then:
```bash
pip install -r lane_detection/requirements.txt
```

## Setup order (run from AKS_Root)
```bash
python -m lane_detection.selftest                 # should print ALL PASS
python -m lane_detection.calibrate_ipm --source oak --save-oak-intrinsics lane_detection/intrinsics.json \
    --config lane_detection/lane_config.json --width-m 0.6 --length-m 0.9 --near-m 0.3
python -m lane_detection.tune_hsv --source oak --config lane_detection/lane_config.json   # s = save
python -m lane_detection.run_lanes --source oak --config lane_detection/lane_config.json
```
Use `--source some_clip.mp4` instead of `oak` to work from a recording.
The scripts also run directly from inside the folder (`python selftest.py`).
Measure the real track width and set `lane_width_m` in `lane_config.json`.

## Using it from other code
```python
from lane_detection import LaneConfig, LaneDetector
det = LaneDetector(LaneConfig.load("lane_detection/lane_config.json"))
res = det.process(frame_bgr, exclude_boxes=yolo_boxes)   # boxes normalized (xmin, ymin, xmax, ymax)
if res.ok:
    res.target_m, res.lateral_offset_m, res.confidence
```

## Output (LaneResult)
Frame: meters on the ground, origin below the camera, y forward, x right. Fits are `x = a*y^2 + b*y + c`.
- `lateral_offset_m`: lane center relative to camera (+ = center is to the right -> steer right)
- `heading_error_rad`: + = lane heads right
- `curvature_1pm`: at lookahead, + = bending right
- `target_m`: point on the center line at `lookahead_m` (feed to pure pursuit)
- `source` / `confidence`: `both` (1.0 max), `left_only` / `right_only` (0.5 max, center = line +/- lane_width/2)

## Gotchas
- Recalibrate IPM whenever the camera mount moves. IPM assumes flat ground; hard accel/braking pitch shifts it.
- Undistort the OAK-D Wide frames (`--save-oak-intrinsics`); otherwise straight tape bends in the BEV. Use a 4:3 size (640x480).
- If the detector pipeline already owns the OAK-D, add a second `requestOutput` to that pipeline instead of opening `sources.py`'s.
- White cones, karts or glare can pass the threshold. Pass YOLO boxes as `exclude_boxes` to blank them out.
- Check that YOLO boxes line up with the lane frame (the detector's 16:9 input vs the 4:3 lane frame may crop differently).
