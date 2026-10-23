# autonomy

The decision-making core for the 1/10 car, and later the full-size kart. It takes obstacles from perception and turns them into steering, throttle and brake commands, under a safety state machine.

Everything except `adapters/` is pure Python with numpy only. There's no camera, GPIO or DepthAI code, so the full stack can be tested on a laptop. Each part class has a `run()` method, so it can be wrapped as a Donkey Car part later.

## Setup and commands

Run these from `AKS_Root\`. The package targets Python 3.10, the Jetson's version.

```powershell
uv venv --python 3.10 autonomy\.venv
uv pip install --python autonomy\.venv\Scripts\python.exe -r autonomy\requirements.txt

autonomy\.venv\Scripts\python -m pytest autonomy\tests -q                  # all tests
autonomy\.venv\Scripts\python -m pytest autonomy\tests\test_supervisor.py -q -k rc_loss   # one test
autonomy\.venv\Scripts\python -m autonomy.replay "$env:AKS_DATA\obstacles.csv" [--show]  # replay + plot
autonomy\.venv\Scripts\python -m autonomy.run --source clip.mp4 [--obstacles obstacles.csv] [--plot run.png]
autonomy\.venv\Scripts\python -m autonomy.run --source oak                                 # live lanes, Ctrl+C stops
```

`autonomy.run` drives the stack from lane detection: offline from a recorded clip in simulated time, or live from the OAK-D. RC and vehicle are mocks, so nothing moves. It needs a calibrated `lane_detection/lane_config.json` (or `--lane-config`); see `lane_detection/README.md`.

Always run as `python -m autonomy...` from `AKS_Root`. Don't run scripts from inside `autonomy\`, because the package's `types.py` would shadow Python's built-in `types` module.

## Conventions

- **Car frame**: meters, x right, y forward, origin on the ground at the camera. Positive `steer_rad` steers right.
- **Commands**: `Command(steer_rad, throttle 0..1, brake 0..1)`. `safety.py` guarantees throttle and brake are never both non-zero.
- **Tunables**: every value is set in `config.yaml`, which loads into frozen dataclasses (`config.py`). A missing or misspelled key raises an error.

## Pipeline (one tick of `loop.py`)

```
perception adapter -> tracker (stub) -> planner -> supervisor -> controller -> safety -> vehicle adapter
                                               ^ RC adapter ----^-----------------------^
```

| Module | Job |
|---|---|
| `planner.py` | Builds the center path, in priority order: cones on both sides (pairs left (x<0) and right cones by distance and uses their midpoints), then the lane center line, then cones on one side only (offset by half of `cone_track_width_m`), then nothing (confidence 0). `lane_path_from_poly()` samples lane_detection's `x = a·y² + b·y + c` fit into a path. |
| `supervisor.py` | Runs the mode state machine below. `step()` is pure; the `Supervisor` class remembers when a FAULT started and how many ticks in a row path confidence has been low. |
| `controller.py` | Pure pursuit steering, `delta = atan(2·L·sin α / Ld)`, measured from the rear axle. Target speed comes from the state. Braking uses `d_stop = v·latency + v²/2a` plus a per-class margin. Only obstacles inside the corridor along the path count. A kart ahead caps speed to keep `follow_distance_m`, and cones are never braked for. |
| `safety.py` | Clamps outputs to limits. URGENT_STOP, FAULT and DISARMED force full brake with zero throttle. MANUAL passes the RC sticks through (negative throttle = brake). |
| `geometry.py` | The corridor and stopping-distance math, shared by the supervisor and controller. |
| `loop.py` | Fixed-rate loop. It warns on overrun and re-anchors if it falls more than a tick behind. If anything ends it abnormally (an exception, Ctrl+C, or SIGTERM turned into `SystemExit`), it sends full brake before re-raising. It collects `warnings` from adapters into the supervisor's `health`, and keeps a bounded in-memory history (`loop.history_max_ticks`). `SimClock` provides simulated time for tests and replay. |
| `paths.py` | `REPO`, `MODELS` (`AKS_MODELS`) and `DATA` (`AKS_DATA`), stdlib only, so any venv can import it. |
| `storage.py` | `DataBudget` stops recording before `$AKS_DATA` exceeds `AKS_DATA_MAX_GB` or the disk drops below `AKS_DATA_MIN_FREE_GB`. It never deletes anything. `setup_logging()` writes a rotating `aks.log`. |

## States

```
DISARMED --arm (switch on MANUAL)--> MANUAL <--RC mode switch--> AUTO <--> SLOW
                                       ^                           |        |
             reset (switch on MANUAL)  |                  stale perception  |
                                       |                           v        |
                                  URGENT_STOP <--held fault_escalate_s-- FAULT
                                       ^
     from any armed state: RC link lost, RC timeout, RC e-stop, software stop request,
     obstacle within stop distance, race control revokes AUTO
```

- **SLOW** applies at once when an obstacle is within `slow_factor` × stop distance. It also applies when path confidence has been below `min_path_confidence` for `low_confidence_ticks` ticks in a row (3, which is 150 ms), so one bad lane frame doesn't slow the car. An empty frame therefore gives SLOW after 3 ticks. Recovery to AUTO is immediate.
- **URGENT_STOP is latched.** It clears only when the mode switch is on MANUAL and reset is pressed, so the car can resume without anyone walking onto the track, as the rulebook requires.
- **RC loss goes straight to URGENT_STOP**, with no FAULT step, because the rulebook requires an Urgent Stop on RC loss.
- **Stale perception is ignored in MANUAL**, since a human is driving.
- **`RcInput.allowed_auto`** is reserved for the competition's Race Control System (RCS) black box. `None` means no black box is attached.
- **Health** (`Supervisor.health`) reports the state, `ok`, and any adapter warnings, such as "recording stopped: disk cap". Warnings never change the state. `ok` is false during FAULT or URGENT_STOP, or while there are any warnings. It's what the RCS Black Box "Health Status" will be fed from.

## Adapters

| Adapter | Status |
|---|---|
| `MockPerception` | Scripted `(t_start, [(cls, x, y[, box]), ...])` segments. A segment of `None` simulates the camera going silent. `cone_gates()` builds test courses. |
| `CsvReplayPerception` | Replays `obstacles.csv` from `oak_obstacles.py`, with `x = csv_lateral_sign · lateral_m` and `y = forward_m`. |
| `LanePerception` | Runs `lane_detection.LaneDetector` on frames from a clip, image, webcam or the OAK-D and returns the lane center as a `Path` (`source="lanes"`). It can wrap another perception adapter: those obstacles pass through, and their detector `box`es are given to the lane detector as `exclude_boxes`. Clip frames are timed by the clip's FPS. |
| `MockRc` | Scripted RC inputs. When the link is lost, the packet timestamp stops advancing. |
| `TickRecorder` | Writes one CSV row per tick to `$AKS_DATA/logs/ticks_*.csv` within the `DataBudget`. At the cap it stops for good and reports a warning; the file is kept. |
| `MockVehicle` | Logs every command and simulates speed: throttle `t` settles at `t · max_speed`. |
| `OakPerception`, `MotorVehicle`, `RcReceiver` | Stubs that raise `NotImplementedError`. |

## Known gaps

- **Unmeasured estimates**: `latency_s` (0.19 s) and `brake_decel_mps2` (3.0 m/s²) come from the progress report and haven't been measured. Stop distances depend directly on them.
- **Perception frame**: perception reports distance along the camera axis, with no correction for the camera's height or pitch. The lateral sign is also untested with a tape measure.
- **Empty frames aren't logged**: `obstacles.csv` has no row for a frame with no detections, so a replay holds the last frame, and a long gap shows as stale data. Logging one heartbeat row per frame in `oak_obstacles.py` would fix this.
- **Tracker**: still a pass-through. One noisy detection can trigger a latched URGENT_STOP.
- **Box alignment**: YOLO boxes are passed to the lane detector as-is. The detector's 16:9 input and a 4:3 lane frame may crop differently, so check alignment before relying on the masking.
- **No boxes in `obstacles.csv`**: the CSV doesn't log detector boxes, so offline runs with `--obstacles` can't mask lanes. Adding box columns to `oak_obstacles.py` would fix this.
- **No path for a long time**: SLOW keeps creeping forward at `slow_speed_mps` with no path. A timeout to a stop may be wanted.
- **One camera, one process**: `--source oak` opens the OAK-D for lanes only. Running lanes and obstacles together needs the real `OakPerception` adapter to share one DepthAI pipeline.
