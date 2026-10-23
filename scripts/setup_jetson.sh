#!/usr/bin/env bash
# Set up a Jetson Orin Nano for the AKS stack. Safe to re-run.
#
#   git clone https://github.com/TJhayes22/usd-aks.git ~/AKS_Root
#   bash ~/AKS_Root/scripts/setup_jetson.sh
#
# Creates:   ~/aks_models  ~/aks_data/{raw,recordings,logs}  ~/venvs/aks
# Installs:  requirements-jetson.txt into the venv, the Luxonis udev rule for the OAK-D (sudo)
# Override any location by exporting AKS_MODELS / AKS_DATA / AKS_VENV / PYTHON before running.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AKS_MODELS="${AKS_MODELS:-$HOME/aks_models}"
AKS_DATA="${AKS_DATA:-$HOME/aks_data}"
AKS_VENV="${AKS_VENV:-$HOME/venvs/aks}"
PYTHON="${PYTHON:-python3}"

echo "== repo: $REPO"

echo "== folders"
mkdir -p "$AKS_MODELS" "$AKS_DATA/raw" "$AKS_DATA/recordings" "$AKS_DATA/logs" "$(dirname "$AKS_VENV")"

echo "== python"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: $PYTHON not found (JetPack 6 ships python3 = 3.10)" >&2
    exit 1
fi
PYVER="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "$PYVER" != "3.10" ]; then
    echo "warning: $PYTHON is $PYVER; the code is tested on 3.10 (set PYTHON=python3.10 if installed)" >&2
fi
if ! "$PYTHON" -c "import ensurepip, venv" >/dev/null 2>&1; then
    echo "   installing python3-venv (sudo)"
    sudo apt-get update && sudo apt-get install -y "python${PYVER}-venv"
fi

echo "== venv: $AKS_VENV"
if [ ! -x "$AKS_VENV/bin/python" ]; then
    "$PYTHON" -m venv "$AKS_VENV"
fi
"$AKS_VENV/bin/python" -m pip install --upgrade pip
"$AKS_VENV/bin/python" -m pip install -r "$REPO/requirements-jetson.txt"

echo "== OAK-D udev rule (sudo)"
RULE_FILE=/etc/udev/rules.d/80-movidius.rules
RULE='SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"'
if [ -f "$RULE_FILE" ] && grep -qF "$RULE" "$RULE_FILE"; then
    echo "   already installed"
else
    echo "$RULE" | sudo tee "$RULE_FILE" >/dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    echo "   installed $RULE_FILE (unplug and replug the OAK-D)"
fi

echo "== check (from a different directory, as the service will)"
(cd /tmp && AKS_MODELS="$AKS_MODELS" AKS_DATA="$AKS_DATA" PYTHONPATH="$REPO" "$AKS_VENV/bin/python" - <<'PY'
import cv2, depthai, numpy, yaml
from autonomy.config import load_config
from autonomy.paths import DATA, MODELS
from autonomy.storage import DataBudget
from lane_detection import LaneDetector
load_config()
s = DataBudget(DATA).check(force=True)
print(f"   depthai {depthai.__version__}, opencv {cv2.__version__}, numpy {numpy.__version__}")
print(f"   MODELS={MODELS}  DATA={DATA}  ({s.used_bytes / 1e9:.2f} GB used, {s.free_bytes / 1e9:.1f} GB free)")
PY
)
if ! ls "$AKS_MODELS"/*.rvc2.tar.xz >/dev/null 2>&1; then
    echo "   note: no .rvc2.tar.xz in $AKS_MODELS yet"
fi

cat <<EOF

Done. Add these lines to ~/.bashrc, then open a new shell:

    export AKS_MODELS="$AKS_MODELS"
    export AKS_DATA="$AKS_DATA"
    export AKS_DATA_MAX_GB=20        # recording stops at this size of \$AKS_DATA (nothing is deleted)
    export AKS_DATA_MIN_FREE_GB=2    # ...or when the disk has this little free space left
    source "$AKS_VENV/bin/activate"

Copy the model from the laptop (PowerShell):

    scp models\\aks_obj_detection.rvc2.tar.xz <user>@<jetson>:$AKS_MODELS/

Run from $REPO (see README "Running on the Jetson"; deploy/aks.service runs it at boot).
EOF
