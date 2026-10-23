"""Nothing may depend on the launch directory, a drive letter, or backslash paths.

Imports and config loads run in a subprocess whose working directory is an empty temp
folder, so module caching from the rest of the test run can't hide a cwd dependency.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from autonomy.paths import REPO

# Every module importable in the autonomy venv (no depthai / ultralytics / supervisely here).
MODULES = [
    "autonomy.paths", "autonomy.types", "autonomy.config", "autonomy.geometry",
    "autonomy.supervisor", "autonomy.planner", "autonomy.controller", "autonomy.safety",
    "autonomy.tracker", "autonomy.loop", "autonomy.replay", "autonomy.run",
    "autonomy.adapters.lane_perception",
    "autonomy.adapters.mock_perception", "autonomy.adapters.mock_rc", "autonomy.adapters.mock_vehicle",
    "autonomy.adapters.csv_replay", "autonomy.adapters.oak_perception", "autonomy.adapters.motor_vehicle",
    "lane_detection", "lane_detection.lane_detector", "lane_detection.sources",
    "lane_detection.run_lanes", "lane_detection.tune_hsv", "lane_detection.calibrate_ipm",
    "lane_detection.selftest",
]

# Scripts run as plain files in other venvs: checked statically.
SKIP_DIRS = {"oak-env", "yolo-env", "ls-env", "oak-examples", "cone_dataset", "fsoco_yolo",
             "fsoco_bounding_boxes_train", "merged", "runs", "raw", "node_modules", "__pycache__"}
DRIVE_PATH = re.compile(r"\b[A-Za-z]:[\\/]+(Users|Program|Windows)")


def _run_elsewhere(tmp_path, code):
    env = dict(os.environ, PYTHONPATH=str(REPO))
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return out.stdout


def _repo_python_files():
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for f in files:
            if f.endswith(".py"):
                yield Path(root) / f


def test_modules_import_from_any_directory(tmp_path):
    code = "import importlib\n" + "".join(f"importlib.import_module({m!r})\n" for m in MODULES)
    _run_elsewhere(tmp_path, code + "print('ok')")


def test_autonomy_config_loads_from_any_directory(tmp_path):
    out = _run_elsewhere(tmp_path, "from autonomy.config import load_config; print(load_config().loop.hz)")
    assert float(out) > 0


@pytest.mark.parametrize("stored", ["intr/intrinsics.json", "intr\\intrinsics.json"])
def test_lane_config_paths_resolve_next_to_the_config(tmp_path, stored):
    cfg_dir = tmp_path / "cfg"
    (cfg_dir / "intr").mkdir(parents=True)
    intr = cfg_dir / "intr" / "intrinsics.json"
    intr.write_text(json.dumps({"K": [[400, 0, 320], [0, 400, 240], [0, 0, 1]],
                                "D": [0, 0, 0, 0, 0], "size": [640, 480]}), encoding="utf-8")
    cfg_file = cfg_dir / "lane_config.json"
    cfg_file.write_text(json.dumps({"intrinsics_path": stored}), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    code = (
        "from pathlib import Path\n"
        "from lane_detection import LaneConfig, LaneDetector\n"
        f"cfg = LaneConfig.load({str(cfg_file)!r})\n"
        f"assert Path(cfg.intrinsics_path) == Path({str(intr)!r}).resolve(), cfg.intrinsics_path\n"
        "LaneDetector(cfg)   # Undistorter opens the intrinsics file\n"
        f"cfg.save({str(cfg_file)!r})\n"
    )
    _run_elsewhere(elsewhere, code)
    saved = json.loads(cfg_file.read_text(encoding="utf-8"))
    assert saved["intrinsics_path"] == "intr/intrinsics.json"   # relative, forward slashes


def test_lane_script_config_defaults_sit_beside_the_package():
    for mod in ("run_lanes", "tune_hsv", "calibrate_ipm"):
        text = (REPO / "lane_detection" / f"{mod}.py").read_text(encoding="utf-8")
        assert 'default=str(DEFAULT_CONFIG)' in text, mod


def test_no_hardcoded_drive_paths():
    me = Path(__file__).resolve()
    hits = []
    for path in _repo_python_files():
        if path.resolve() == me:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if DRIVE_PATH.search(line):
                hits.append(f"{path.relative_to(REPO)}:{i}: {line.strip()}")
    assert not hits, "absolute Windows paths:\n" + "\n".join(hits)


def test_script_bootstraps_point_at_the_repo_root():
    """Scripts outside the package add parents[N] to sys.path; N must land on the repo root."""
    pattern = re.compile(r"sys\.path\.append\(str\(Path\(__file__\)\.resolve\(\)\.parents\[(\d)\]\)\)")
    found = 0
    for path in _repo_python_files():
        for m in pattern.finditer(path.read_text(encoding="utf-8", errors="replace")):
            found += 1
            assert path.resolve().parents[int(m.group(1))] == REPO, path
    assert found >= 3


def test_env_vars_override_models_and_data(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO), AKS_MODELS=str(tmp_path / "m"), AKS_DATA=str(tmp_path / "d"))
    out = subprocess.run([sys.executable, "-c", "from autonomy.paths import MODELS, DATA; print(MODELS); print(DATA)"],
                         cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    models, data = out.stdout.split()
    assert Path(models) == (tmp_path / "m").resolve() and Path(data) == (tmp_path / "d").resolve()
