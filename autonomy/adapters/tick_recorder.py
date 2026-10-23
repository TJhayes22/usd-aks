"""Write one CSV row per loop tick to $AKS_DATA/logs, within the disk budget.

When the budget runs out, recording stops for good (the file is kept, nothing is deleted)
and `warnings` reports why; the loop passes that to the supervisor's health status. Driving
is not affected.
"""
import csv
import io
import logging
from pathlib import Path
from typing import Optional, Tuple

from ..storage import DataBudget

log = logging.getLogger(__name__)
COLUMNS = ["t", "state", "steer_rad", "throttle", "brake", "speed_mps", "n_obstacles", "path_confidence", "warnings"]


class TickRecorder:
    def __init__(self, path: Path, budget: DataBudget):
        self.path = Path(path)
        self.budget = budget
        self.rows = 0
        self.stopped_reason: Optional[str] = None
        self._file = None
        self._writer = None

    @property
    def warnings(self) -> Tuple[str, ...]:
        return (f"recording stopped: {self.stopped_reason}",) if self.stopped_reason else ()

    def _stop(self, reason: str) -> None:
        self.stopped_reason = reason
        log.warning("tick recording stopped after %d rows (%s): %s", self.rows, self.path, reason)
        self.close()

    def run(self, entry) -> None:
        if self.stopped_reason:
            return
        status = self.budget.check()
        if not status.ok:
            self._stop(status.reason)
            return
        row = [f"{entry.t:.3f}", entry.state.name, f"{entry.cmd.steer_rad:.4f}", f"{entry.cmd.throttle:.3f}",
               f"{entry.cmd.brake:.3f}", f"{entry.speed_mps:.3f}", entry.n_obstacles,
               f"{entry.path_confidence:.2f}", "; ".join(entry.warnings)]
        line = io.StringIO()
        csv.writer(line).writerow(row)
        try:
            if self._file is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._file = open(self.path, "w", newline="", encoding="utf-8")
                self._writer = csv.writer(self._file)
                self._writer.writerow(COLUMNS)
            self._file.write(line.getvalue())
            self._file.flush()
        except OSError as e:            # e.g. disk actually full: stop recording, keep driving
            self._stop(f"write failed: {e}")
            return
        self.budget.add(len(line.getvalue()))
        self.rows += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
