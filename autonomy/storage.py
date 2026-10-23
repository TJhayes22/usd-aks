"""Disk budget for $AKS_DATA, and log setup. Stdlib only, so every venv can import it.

Recording stops (with a warning) before the SSD fills; nothing is ever deleted. Hitting the
cap only stops recording: callers keep running, and the warning reaches the logs and, inside
the driving stack, the supervisor's health status.

Limits are deployment settings, so they come from environment variables like AKS_DATA does:
    AKS_DATA_MAX_GB       total size of $AKS_DATA before recording stops   (default 20)
    AKS_DATA_MIN_FREE_GB  free space to leave on the disk                  (default 2)
"""
import logging
import logging.handlers
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .paths import DATA

GB = 1024 ** 3
DEFAULT_MAX_DATA_GB = 20.0
DEFAULT_MIN_FREE_GB = 2.0
RESCAN_S = 30.0               # full folder walk at most this often; writes in between are counted
LOG_MAX_BYTES = 10 * 1024 ** 2
LOG_BACKUPS = 5               # aks.log + 5 rotated files = at most 60 MB of logs

log = logging.getLogger(__name__)


def _env_gb(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def fmt_bytes(n: float) -> str:
    return f"{n / GB:.1f} GB" if abs(n) >= GB else f"{n / 1024 ** 2:.1f} MB"


def dir_size(root: Path) -> int:
    """Total bytes of regular files under root (unreadable entries are skipped)."""
    total, stack = 0, [Path(root)]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except OSError:
            continue
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    stack.append(Path(e.path))
                elif e.is_file(follow_symlinks=False):
                    total += e.stat(follow_symlinks=False).st_size
            except OSError:
                pass
    return total


@dataclass(frozen=True)
class StorageStatus:
    ok: bool
    used_bytes: int
    free_bytes: int
    reason: str = ""


class DataBudget:
    """Answers "may I write more?" for a data folder, without walking it on every call."""

    def __init__(self, root: Path = DATA, max_bytes: Optional[int] = None, min_free_bytes: Optional[int] = None,
                 clock: Callable[[], float] = time.monotonic, rescan_s: float = RESCAN_S):
        self.root = Path(root)
        self.max_bytes = int(_env_gb("AKS_DATA_MAX_GB", DEFAULT_MAX_DATA_GB) * GB) if max_bytes is None else max_bytes
        self.min_free_bytes = (int(_env_gb("AKS_DATA_MIN_FREE_GB", DEFAULT_MIN_FREE_GB) * GB)
                               if min_free_bytes is None else min_free_bytes)
        self.clock, self.rescan_s = clock, rescan_s
        self._used = 0
        self._added = 0            # bytes reported via add() since the last walk
        self._last_scan: Optional[float] = None

    def add(self, nbytes: int) -> None:
        self._added += max(int(nbytes), 0)

    def check(self, force: bool = False) -> StorageStatus:
        now = self.clock()
        if force or self._last_scan is None or now - self._last_scan >= self.rescan_s:
            self.root.mkdir(parents=True, exist_ok=True)
            self._used, self._added, self._last_scan = dir_size(self.root), 0, now
        used = self._used + self._added
        free = shutil.disk_usage(self.root).free - self._added
        if used >= self.max_bytes:
            return StorageStatus(False, used, free,
                                 f"{self.root} holds {fmt_bytes(used)}, cap is {fmt_bytes(self.max_bytes)} (AKS_DATA_MAX_GB)")
        if free <= self.min_free_bytes:
            return StorageStatus(False, used, free,
                                 f"only {fmt_bytes(free)} free on disk, keeping {fmt_bytes(self.min_free_bytes)} "
                                 "(AKS_DATA_MIN_FREE_GB)")
        return StorageStatus(True, used, free)


def setup_logging(log_dir: Path = DATA / "logs", name: str = "aks.log", level: int = logging.INFO) -> Path:
    """Console + size-capped rotating file log. Returns the log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / name
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    for handler in (logging.StreamHandler(),
                    logging.handlers.RotatingFileHandler(path, maxBytes=LOG_MAX_BYTES,
                                                         backupCount=LOG_BACKUPS, encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    return path
