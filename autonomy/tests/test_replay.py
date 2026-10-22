from autonomy.replay import plot, replay
from autonomy.types import State


def test_replay_runs_and_plots(cfg, tmp_path):
    csv = tmp_path / "obstacles.csv"
    rows = ["ts,label,conf,forward_m,lateral_m"]
    rows += [f"{100 + i * 0.1:.1f},person,0.7,6.0,2.0" for i in range(10)]      # off to the side
    rows += [f"{101 + i * 0.1:.1f},person,0.7,1.0,0.0" for i in range(5)]       # steps in front
    csv.write_text("\n".join(rows) + "\n")

    history = replay(csv, cfg, tail_s=0.5)
    seen = {h.state for h in history}
    assert State.SLOW in seen and history[-1].state == State.URGENT_STOP

    out = tmp_path / "replay.png"
    plot(history, out, show=False)
    assert out.stat().st_size > 0
