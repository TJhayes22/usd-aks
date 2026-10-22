"""Core logic must stay hardware-free: only stdlib, numpy, and package-local imports."""
import ast
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
CORE = ["types.py", "config.py", "geometry.py", "supervisor.py", "planner.py",
        "controller.py", "safety.py", "tracker.py", "loop.py"]
ALLOWED_THIRD_PARTY = {"numpy"}
EXTRA_ALLOWED = {"config.py": {"yaml"}}   # config loading needs a YAML parser


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module.split(".")[0]


@pytest.mark.parametrize("name", CORE)
def test_core_module_is_hardware_free(name):
    path = PKG / name
    if not path.exists():
        pytest.skip(f"{name} not written yet")
    allowed = ALLOWED_THIRD_PARTY | EXTRA_ALLOWED.get(name, set())
    bad = [m for m in _imports(path) if m not in sys.stdlib_module_names and m not in allowed]
    assert not bad, f"{name} imports {bad}; hardware and I/O libraries belong in adapters/"
