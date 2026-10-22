"""Shared filesystem locations, the same on the Windows laptop and the Jetson.

    REPO    the git checkout (resolved from this file, so the launch directory doesn't matter)
    MODELS  model weights and .rvc2.tar.xz archives: env AKS_MODELS, default ~/aks_models
    DATA    recordings, logs, obstacles.csv:         env AKS_DATA,   default ~/aks_data

Weights and data live outside the repo and out of git. Stdlib only, so every venv
(oak-env, yolo-env, autonomy) can import it.

Scripts outside this package (oak-template/, data_collection/, ...) run as plain files,
so they add the repo root to sys.path first:

    sys.path.append(str(Path(__file__).resolve().parents[1]))   # parents[2] from one level deeper
    from autonomy.paths import DATA, MODELS
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _env_dir(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.expanduser()


MODELS = _env_dir("AKS_MODELS", Path.home() / "aks_models")
DATA = _env_dir("AKS_DATA", Path.home() / "aks_data")
