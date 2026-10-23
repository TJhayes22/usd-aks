# AKS Perception Pipeline

Perception software for USD's senior capstone entry in the Autonomous Karting Series (AKS). The team is building a 1/10-scale autonomous car first, and the software carries over to the full-size electric kart (target speed around 30 mph).

This repo holds multi-class object detection (cone, person, kart) on an OAK-D, plus tools to collect and label data, train the model, and turn detections into an obstacle list for avoidance.

**Classes (fixed order):** `0 cone`, `1 person`, `2 kart` (RC cars are labeled as `kart`).

## Hardware

- **Compute:** Jetson Orin Nano. Development and training happen on a Windows PC for now.
- **Camera:** OAK-D Wide Pro on DepthAI v3. YOLO runs on the camera itself from a Luxonis HubAI `.rvc2.tar.xz` archive.
- **Planned:** 3D lidar and GNSS/INS.

## Rulebook constraints on the software

- All computation runs on the vehicle. Nothing is computed offboard while it's running.
- Software must be able to command an Urgent Stop, and the Urgent Stop must trigger if the RC controller goes out of range or disconnects.
- The car must support RC driving and switching between RC and autonomous modes.
- A mechanical brake is required. Braking with the motor alone doesn't count.

## Conventions

- **Car frame:** meters, x to the right, y forward, origin on the ground under the camera. `oak_obstacles.py` doesn't follow this yet: it reports `forward_m`/`lateral_m` along the camera axis, with no height or pitch correction.
- Pure logic stays hardware-free, and hardware access goes only in `adapters/` (not created yet).
- New code must run on Python 3.10, the Jetson's version, even though the Windows venvs use 3.12.
- Donkey Car is the likely vehicle framework, so structure components as parts with `run()` methods.

---

## Repo layout

Paths below are relative to `AKS_Root\`. Adjust if your folders differ.

```
AKS_Root\
  models\                          training + exported models
    train.py                       train from COCO or cone-only weights
    finetune.py                    fine-tune the best model on merged\, compare old vs new
    merged\                        merged cone/person/kart dataset (data.yaml)
    yolo-env\                      training venv
  oak-template\                    OAK-D pipelines (based on the Luxonis template)
    main.py                        viewer pipeline (browser visualizer)
    oak_obstacles.py               detections + stereo depth -> obstacle list
  data_collection\
    scripts\record_frames.py       save OAK-D frames for labeling (to $AKS_DATA\raw)
    prelabel.py                    pre-label frames for Label Studio (YOLO-World or own .pt)
    merge_labelstudio.py           merge a Label Studio export into the dataset, split by session
    raw\<session>\                 recorded frames, one folder per session (laptop: AKS_DATA points here)
    exports\                       Label Studio YOLO exports
  fsoco_work\                      FSOCO -> YOLO conversion (convert.py, prepare_dataset.py)
  cone_dataset\                    single-class cone dataset built from FSOCO
  autonomy\                        decision-making core: supervisor, planner, controller, safety (see autonomy\README.md)
    paths.py                       shared REPO / MODELS / DATA locations
  lane_detection\                  tape-line lane detection (see lane_detection\README.md)
  oak-examples\                    upstream Luxonis examples, for reference
  requirements.txt                 oak-env dependencies
  .env                             DEPTHAI_HUB_API_KEY (never commit)
```

### Model and data folders

Weights and runtime data live outside the repo and out of git. `autonomy\paths.py` resolves them the same way on every machine, so scripts work from any working directory:

| Name | Environment variable | Default | Holds |
|---|---|---|---|
| `MODELS` | `AKS_MODELS` | `~/aks_models` | `.rvc2.tar.xz` archives the camera scripts load |
| `DATA` | `AKS_DATA` | `~/aks_data` | `raw\` recorded frames, `obstacles.csv`, recordings, logs |

On the laptop, keep labeling sessions in `data_collection\raw\` by pointing `AKS_DATA` there, and copy the model archive into `~/aks_models` (or point `AKS_MODELS` at `models\`):

```powershell
[Environment]::SetEnvironmentVariable("AKS_DATA", "C:\Users\Tyler\AKS_Root\data_collection", "User")
New-Item -ItemType Directory -Force "$HOME\aks_models"
Copy-Item models\aks_obj_detection.rvc2.tar.xz "$HOME\aks_models\"
```

Open a new terminal afterwards so the variable is picked up.

---

## 1. Python environments

Use **three separate virtual environments** on **Python 3.12**. Do not install packages into the global Python, and avoid 3.14: NumPy crashed on import there (`OverflowError: cannot convert longdouble infinity to integer`). Always check that your prompt shows `(env-name)` before running `pip install`.

Install Python 3.12 if needed:

```powershell
py -0                                          # list installed versions
winget install -e --id Python.Python.3.12      # if 3.12 is missing; reopen PowerShell after
```

### oak-env: camera scripts (OAK-D)

```powershell
cd C:\Users\Tyler\AKS_Root
py -3.12 -m venv oak-env
oak-env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import depthai, cv2, numpy; print('ok', depthai.__version__)"
```

`requirements.txt` pins `depthai`, `depthai-nodes`, `numpy`, `opencv-python`, and `python-dotenv`.

Use `opencv-python`, not `opencv-python-headless`: the headless build has no preview windows. The current `requirements.txt` also pins `opencv-python-headless`, so delete that line before installing from it. When you re-lock versions, write the file as UTF-8. In Windows PowerShell 5.1, `pip freeze > requirements.txt` saves it as UTF-16, so use `pip freeze | Out-File -Encoding utf8 requirements.txt` instead.

### yolo-env: training (NVIDIA GPU)

```powershell
py -3.12 -m venv models\yolo-env
models\yolo-env\Scripts\Activate.ps1
# Install PyTorch with CUDA using the command from pytorch.org (pick Windows / Pip / newest CUDA), then:
pip install ultralytics
python -c "import torch; print(torch.cuda.is_available())"     # must print True
```

If that prints `False`, you have the CPU build of PyTorch: uninstall `torch torchvision torchaudio` and reinstall with the CUDA index URL from pytorch.org.

### ls-env: Label Studio

```powershell
cd data_collection
py -3.12 -m venv ls-env
ls-env\Scripts\Activate.ps1
pip install label-studio
label-studio start                             # opens http://localhost:8080
```

Keep Label Studio in its own environment. It pulls in many dependencies that clash with the others.

---

## 2. Run the camera pipelines (oak-env)

### Viewer: `main.py`

```powershell
oak-env\Scripts\Activate.ps1
cd oak-template
python main.py
```

Open http://localhost:8082 in a browser. `MODEL_ARCHIVE` at the top of the file is `$AKS_MODELS\aks_obj_detection.rvc2.tar.xz`.

The snaps lines (`SnapsProducer` / `SnapsUploader`) are commented out: with them enabled, the process crashed natively as soon as a person appeared in view. To use them later, put `DEPTHAI_HUB_API_KEY=...` in `.env` in the workspace root first.

### Obstacle list: `oak_obstacles.py`

Runs your model plus stereo depth on the camera and prints one line per detection:

```powershell
python oak_obstacles.py              # headless: prints + writes $AKS_DATA\obstacles.csv
python oak_obstacles.py --show       # also opens a preview window (q to quit)
python oak_obstacles.py --model other.rvc2.tar.xz --log run1.csv   # override either path
```

Output: `class  conf  fwd=<m>  lat=<m>`, nearest first.

Settings are in the CONFIG block at the top of the file:

| Setting | Meaning |
|---|---|
| `MODEL_ARCHIVE` | default `.rvc2.tar.xz` in `$AKS_MODELS` (`--model` overrides) |
| `NN_SHAVES` | compute cores for the network (7 works; try 6 if you get a shaves error) |
| `CONF` | detection confidence threshold |
| `DEPTH_MIN_MM` / `DEPTH_MAX_MM` | valid stereo depth range |
| `LATERAL_SIGN` | set to -1 to flip left/right |

Notes:
- `forward` is the distance along the camera axis, not corrected for camera pitch or height yet.
- Stereo depth is unreliable very close (roughly under 0.7 m) and far away. Test it with a tape measure before trusting it.
- The camera can only be used by one program at a time. Close any other script first.

---

## 3. Collect and label data

### Record frames

```powershell
oak-env\Scripts\Activate.ps1
cd data_collection\scripts
python record_frames.py --session 2026-10-20_lot_sunny --preview
```

Frames save to `$AKS_DATA\raw\<session>\`, which is `data_collection\raw\<session>\` when `AKS_DATA` is set as above. Options:

| Flag | Meaning |
|---|---|
| `--interval 0.5` | seconds between saved frames |
| `--size 1280x720` | saved resolution (16:9 matches the deployed 512x288 shape) |
| `--min-sharp 60` | skip blurry frames (0 = off) |
| `--min-change 2.0` | skip near-duplicate frames (0 = off) |
| `--max-frames N` | stop after N saved frames |
| `--out PATH` | output base folder |

Preview keys: `s` save now, `p` pause, `q` quit. Name sessions `date_place_conditions` so train/val can be split by whole session.

Capture RC cars, karts, people, and close and far cones from the kart's real camera height, in different lighting. Compare one saved frame to the viewer to confirm the field of view matches.

### Pre-label (optional)

`prelabel.py` runs a model over a folder of frames and writes `tasks.json` with predicted boxes for Label Studio to import, so you fix boxes instead of drawing them. It uses YOLO-World zero-shot with text prompts by default, or your own `.pt` with `--keep-classes`. Run it in an isolated environment from `data_collection\`:

```powershell
uv run --python 3.12 --with ultralytics prelabel.py --images to_label
```

`--doc-root` must match Label Studio's `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT`. The confidence default is deliberately low (0.15), since deleting a wrong box is faster than drawing a missing one.

### Label in Label Studio

1. Start it from `ls-env` (see above) and create a project with the **Object Detection with Bounding Boxes** template.
2. Define labels in this order: `cone`, `person`, `kart`.
3. Label every object of all three classes. Unlabeled objects are trained as background.
4. Export as **YOLO** to `data_collection\exports\` and check `classes.txt` reads cone, person, kart.
5. Split train/val by session, not by random frame. Keep a few whole sessions as a test set.

### Merge into the dataset

`merge_labelstudio.py` remaps the export's classes to our IDs and copies frames into the dataset with an `own_` prefix. It sends whole sessions to val:

```powershell
python merge_labelstudio.py --src ls_export --dst ..\models\merged --map rc_car=2 `
    --session-regex '(\d{4}-\d{2}-\d{2}_\d{4})_' --val-sessions 2026-10-20_1531
```

Frames with no boxes are kept as negatives unless you pass `--no-empty`. `finetune.py` uses the `own_` prefix to build an RC-only validation set.

---

## 4. Build the dataset and train

### Build (Google Colab recommended)

`build_dataset.py` merges FSOCO cones, ~4k COCO person images, and the Roboflow kart set, then pseudo-labels the classes each source is missing. It lives in Colab, not in this folder, and its paths assume Colab (`/content/...`). The cone-only input comes from `fsoco_work\convert.py` (FSOCO Supervisely → YOLO, run in `fsoco_work\.venv`) followed by `prepare_dataset.py`, which collapses all cone types to class 0 in `cone_dataset\`.

```python
!pip install -q ultralytics fiftyone roboflow
```

Set `CONE_DS`, `CONE_MODEL`, and `ROBOFLOW_API_KEY` in the CONFIG block, then run it. Spot-check pseudo-labels before training. Do not commit your Roboflow key.

### Train (local NVIDIA GPU)

Unzip the merged dataset to `models\merged` and fix the first line of `merged\data.yaml` to your real path (forward slashes):

```yaml
path: C:/Users/Tyler/AKS_Root/models/merged
```

```powershell
models\yolo-env\Scripts\Activate.ps1
cd models
python train.py coco       # start from COCO weights (recommended)
python train.py cone       # start from the cone-only weights (ablation)
```

The best weights are saved to `runs\detect\<run name>\weights\best.pt`. Copy `best.pt` somewhere safe.

To resume an interrupted run: `YOLO(r"runs\detect\<run>\weights\last.pt").train(resume=True)`.

### Fine-tune on your own data

```powershell
cd models
python finetune.py
```

This script starts from `OLD_WEIGHTS` (currently `runs\detect\cone_person_kart_from_coco-3\weights\best.pt`) and fine-tunes for 40 epochs as run `rc_finetune`. It then prints mAP for the old and new models on the full val set and on the RC-only val set. It exits early if there's no CUDA GPU or if `data.yaml`'s `path:` doesn't point to `models\merged`.

### Export to the OAK-D

1. Upload `best.pt` to Luxonis HubAI and convert for RVC2.
2. Input size is chosen at conversion, not in code. 640x640 ran at ~7 FPS in the viewer; 512x288 ran at ~17 FPS in the viewer (network only, before stereo depth was added).
3. Download the `.rvc2.tar.xz`, rename it, and put it in `$AKS_MODELS` (or update `MODEL_ARCHIVE`). Class names come from the archive. Don't extract it.

---

## 5. Autonomy stack

`autonomy\` turns the obstacle list into steering, throttle and brake commands under a safety state machine (DISARMED, MANUAL, AUTO, SLOW, URGENT_STOP, FAULT). The logic is hardware-free and runs on Python 3.10 for the Jetson. All tunables are in `autonomy\config.yaml`.

```powershell
uv venv --python 3.10 autonomy\.venv
uv pip install --python autonomy\.venv\Scripts\python.exe -r autonomy\requirements.txt
autonomy\.venv\Scripts\python -m pytest autonomy\tests -q
autonomy\.venv\Scripts\python -m autonomy.replay "$env:AKS_DATA\obstacles.csv" --show
```

The replay command runs a recorded `obstacles.csv` through the whole stack in simulated time and saves a plot of steering, throttle/brake and state. Real camera, motor and RC adapters are still stubs. See `autonomy\README.md` for the design and the known gaps.

Lane detection can drive the stack too, offline from a recorded clip or live from the OAK-D. RC and vehicle are mocks, so nothing moves:

```powershell
autonomy\.venv\Scripts\python -m autonomy.run --source clip.mp4 --plot run.png
autonomy\.venv\Scripts\python -m autonomy.run --source oak
```

It needs a calibrated `lane_detection\lane_config.json` (see `lane_detection\README.md`).

---

## 6. Running on the Jetson

The Jetson Orin Nano runs the same code from a git clone. Weights and data stay outside the clone:

| Path | Holds |
|---|---|
| `~/AKS_Root` | git clone, code only |
| `~/aks_models` | `.rvc2.tar.xz` model archives (`AKS_MODELS`), copied from the laptop |
| `~/aks_data` | `raw/` frames, `recordings/` clips, `logs/` (`aks.log`, `ticks_*.csv`), `obstacles.csv` (`AKS_DATA`) |
| `~/venvs/aks` | one Python 3.10 venv for the camera, lanes and autonomy (`requirements-jetson.txt`) |

### Setup (once, safe to re-run)

```bash
git clone https://github.com/TJhayes22/usd-aks.git ~/AKS_Root
bash ~/AKS_Root/scripts/setup_jetson.sh
```

The script:
1. Creates the folders above.
2. Builds `~/venvs/aks` from `requirements-jetson.txt`, which installs DepthAI 3.11.0 (the same as the laptop).
3. Installs the Luxonis udev rule so the OAK-D works without root. Unplug and replug the camera afterwards.
4. Checks the install from a different directory.
5. Prints the `export` lines to add to `~/.bashrc`.

Then copy the model over from the laptop:

```powershell
scp models\aks_obj_detection.rvc2.tar.xz <user>@<jetson>:~/aks_models/
```

`requirements-jetson.txt` pins `numpy==2.2.6`, not the laptop's 2.5.4, because numpy 2.3 and later need Python 3.11+. It also leaves out `depthai-nodes`, which is only used by `main.py`'s browser viewer. That package pulls in `opencv-python-headless`, which breaks preview windows; the file explains how to add it if you need the viewer.

### Running

From `~/AKS_Root`, with the venv active:

```bash
python -m autonomy.run --source oak                         # lanes from the OAK-D, mock RC/vehicle
python -m autonomy.run --source ~/aks_data/recordings/run1.mp4
python oak-template/oak_obstacles.py                        # obstacle list -> ~/aks_data/obstacles.csv
```

Only one program can hold the OAK-D at a time.

### Disk space

Recording stops before the SSD fills, and nothing is ever deleted. The limits are set by environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `AKS_DATA_MAX_GB` | 20 | Recording stops when `$AKS_DATA` reaches this size |
| `AKS_DATA_MIN_FREE_GB` | 2 | Recording also stops when the disk has this little free space left |

Hitting a limit **only stops recording**: tick logs, `obstacles.csv` rows and `record_frames.py` frames. The driving stack keeps running normally. The warning appears in `aks.log`, the console and the supervisor's health status (`health: NOT ok`, with the reason), but it never changes the driving state.

`aks.log` itself rotates at 10 MB and keeps 5 old files, so logs stay under 60 MB. To free space, copy recordings off the Jetson and delete them yourself.

### Start at boot (optional)

`deploy/aks.service` is a systemd user service. It isn't enabled by default. It runs `python -m autonomy.run --source oak` from `~/AKS_Root`, with `AKS_MODELS`, `AKS_DATA` and the disk limits set, and restarts it if it crashes.

```bash
mkdir -p ~/.config/systemd/user
cp ~/AKS_Root/deploy/aks.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now aks.service
sudo loginctl enable-linger "$USER"          # start at boot without logging in
journalctl --user -u aks -f                  # watch it
systemctl --user disable --now aks.service   # turn it off again
```

Stopping the service sends SIGTERM, which makes the loop send full brake before exiting. After a restart, the stack comes up DISARMED.

**Before this drives real motors**, replace the mock RC and vehicle with the real RC receiver and motor adapters, and remove the mock RC's auto-arm in `autonomy/run.py`. Today the service only exercises perception and decisions.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `OverflowError: cannot convert longdouble infinity to integer` on `import numpy` | Python 3.14 environment. Use a Python 3.12 venv. |
| `No module named depthai` | Wrong environment active. Activate `oak-env`. |
| `cv2.error: ... function is not implemented` on `imshow` | Headless OpenCV. `pip uninstall -y opencv-python opencv-python-headless`, then `pip install opencv-python`. |
| `main.py` dies when a person appears | Snaps uploader without an API key. Keep it commented out or set `DEPTHAI_HUB_API_KEY`. |
| `Blob compiled for 8 shaves, but only 7 are available` | Stereo depth uses some compute cores. Set `NN_SHAVES = 7` (or 6) in `oak_obstacles.py`. |
| `getConnectedCameras(): incompatible function arguments` | Older DepthAI build. Fixed in the current script. |
| `Dataset ... images not found, missing path C:\content\...` | `data.yaml` still has the Colab path. Edit its `path:` line. |
| Camera not found | `python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"`. Check the USB 3 cable and close other programs using it. |
| CUDA out of memory during training | Lower `batch` in `train.py`. |
| `Model archive not found: ~/aks_models/...` | Copy the `.rvc2.tar.xz` into `$AKS_MODELS`, set `AKS_MODELS`, or pass `--model`. |
| `numpy==2.5.4` won't install on the Jetson | numpy 2.3+ needs Python 3.11+. Install from `requirements-jetson.txt` (numpy 2.2.6). |
| `$'\r': command not found` running `setup_jetson.sh` | Windows line endings. `.gitattributes` keeps `*.sh` LF on checkout; if you copied the file by hand, run `sed -i 's/\r$//' scripts/setup_jetson.sh`. |
| OAK-D not found on the Jetson, works with `sudo` | udev rule missing. Re-run `setup_jetson.sh`, then unplug and replug the camera. |
| `recording stopped: ... cap is ...` | `$AKS_DATA` reached `AKS_DATA_MAX_GB` (or the disk is nearly full). Copy data off, delete it, or raise the limit. Driving isn't affected. |

---

## 8. Security and git

- Never commit `.env`, API keys, or environment folders.
- Model weights (`*.pt`, `*.rvc2.tar.xz`), datasets, recordings and logs are gitignored. Keep them in `$AKS_MODELS` / `$AKS_DATA`.
- Back up raw frames and label exports to Drive. The labeling work can't be regenerated.

---

## 9. Status and next steps

Working:
- YOLO11n 3-class detector, about 17 FPS at 512x288 on the OAK-D (network only)
- Model export
- Viewer
- Obstacle list with stereo depth
- Frame recording
- Own RC-car data labeled and merged (`own_*` images), plus the `rc_finetune` run
- Autonomy core (`autonomy\`): supervisor, cone/lane planner, pure pursuit controller, safety layer, fixed-rate loop, mock and CSV-replay adapters, and tests
- Lane detection (`lane_detection\`) wired into the autonomy stack as a second path source, runnable offline from clips
- Portable paths (`AKS_MODELS` / `AKS_DATA`) and a Jetson setup script, disk cap, and optional systemd service

Not done yet:
- Reach 25+ FPS, and measure real FPS with stereo depth running.
- Test distance and lateral-sign accuracy with a tape measure, and convert obstacles to the car frame (x right, y forward, origin on the ground under the camera).
- Collect more of your own data, retrain, and re-run the range test.
- Calibrate lane detection on the final camera mount and tune the tape color.
- Real adapters for the autonomy stack (OAK-D perception sharing one pipeline for detections and lanes, motors and mechanical brake, RC receiver), a real tracker, and measuring latency and braking deceleration.
- Lidar-camera fusion. The camera script is written so only the distance source needs to change.
- First run on the Jetson hardware: confirm the JetPack version ships Python 3.10, then measure FPS there.