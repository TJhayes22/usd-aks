"""Scripted RC transmitter. Script: list of (t_start, dict of RcInput fields except t).

While link_ok is False the packet timestamp stops advancing, like a real receiver losing the link.
"""
from typing import Dict, List, Tuple

from ..types import RcInput


class MockRc:
    def __init__(self, script: List[Tuple[float, Dict]]):
        self.script = sorted(script, key=lambda s: s[0])
        self.last_rx = float("-inf")

    def run(self, now: float) -> RcInput:
        active = [s for s in self.script if s[0] <= now]
        fields = dict(active[-1][1]) if active else {"link_ok": False}
        fields.setdefault("link_ok", True)
        if fields["link_ok"]:
            self.last_rx = now
        return RcInput(t=self.last_rx, **fields)


def arm_then_auto(t_auto: float = 0.2) -> List[Tuple[float, Dict]]:
    """Arm in MANUAL at t=0, flip the mode switch to AUTO at t_auto."""
    return [(0.0, {"arm": True, "mode_auto": False}), (t_auto, {"mode_auto": True})]
