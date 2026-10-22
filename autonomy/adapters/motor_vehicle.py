"""Real actuator and RC receiver adapters (stubs).

MotorVehicle: map Command to steering servo / ESC / mechanical brake outputs and report speed
(wheel encoder or ESC telemetry). RcReceiver: read the long-range RC receiver (e.g. ELRS) and
report link status, mode switch, arm, reset and e-stop channels as an RcInput.
"""
from ..config import Config
from ..types import Command, RcInput


class MotorVehicle:
    speed_mps = 0.0

    def __init__(self, cfg: Config):
        raise NotImplementedError("MotorVehicle is a stub; use MockVehicle")

    def run(self, cmd: Command, now: float) -> None:
        raise NotImplementedError


class RcReceiver:
    def __init__(self, cfg: Config):
        raise NotImplementedError("RcReceiver is a stub; use MockRc")

    def run(self, now: float) -> RcInput:
        raise NotImplementedError
