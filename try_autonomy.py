"""
Try the autonomy stack on a scripted scenario, in simulated time (no hardware needed).

Run from AKS_Root with the autonomy venv active:
    python try_autonomy.py

Edit the SCENARIO block, predict what will happen, then run it. Ideas:
  ("person", 1.0, 2.5)                 person outside the corridor -> keeps driving
  ("kart", 0.0, 3.0)                   kart ahead -> slows to keep 1.5 m following distance
  cone_gates(x_offset=0.5)             course shifted right -> positive (right) steering
  (2.0, None) as a perception segment  camera goes silent -> FAULT, then URGENT_STOP 0.2 s later
  (2.0, {"link_ok": False}) in RC      RC link drops -> URGENT_STOP immediately
  stop_margin_m: person in autonomy/config.yaml changes when the person triggers SLOW / URGENT_STOP
"""
from autonomy.adapters.mock_perception import MockPerception, cone_gates
from autonomy.adapters.mock_rc import MockRc, arm_then_auto
from autonomy.adapters.mock_vehicle import MockVehicle
from autonomy.config import load_config
from autonomy.loop import Loop, SimClock

# ---------------- SCENARIO ----------------
# Perception: (start time s, list of (class, x m right, y m ahead)); None = camera goes silent
PERCEPTION = [
    (0.0, cone_gates()),                               # straight cone course
    (3.0, cone_gates() + [("person", 0.0, 2.5)]),      # person steps in 2.5 m ahead at 3 s
]
# RC: arm in MANUAL at 0 s, flip the mode switch to AUTO at 0.2 s; add (t, {...}) entries to change it
RC = arm_then_auto()
DURATION_S = 5.0
PRINT_EVERY = 5                                        # ticks between printed rows (loop runs at 20 Hz)
# ------------------------------------------


def main():
    cfg = load_config()
    clock = SimClock()
    loop = Loop(cfg, MockPerception(PERCEPTION), MockRc(RC), MockVehicle(cfg),
                clock=clock, sleep=clock.sleep)
    history = loop.run(duration_s=DURATION_S)

    print("    t  state         speed   steer  throttle  brake")
    prev = None
    for i, h in enumerate(history):
        changed = h.state != prev
        if changed or i % PRINT_EVERY == 0:
            mark = "  <- " + h.state.name if changed and prev is not None else ""
            print(f"{h.t:5.2f}  {h.state.name:12s} {h.speed_mps:5.2f}  {h.cmd.steer_rad:+6.2f}"
                  f"  {h.cmd.throttle:8.2f}  {h.cmd.brake:5.2f}{mark}")
        prev = h.state


if __name__ == "__main__":
    main()
