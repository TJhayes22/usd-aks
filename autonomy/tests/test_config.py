import pytest
import yaml

from autonomy.config import DEFAULT_PATH, from_dict


def _raw():
    with open(DEFAULT_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_default_config_loads(cfg):
    assert cfg.loop.hz > 0
    assert cfg.obstacles.margin("person") > cfg.obstacles.margin("kart")
    assert cfg.obstacles.margin("dog") == cfg.obstacles.stop_margin_m["default"]


def test_unknown_key_rejected():
    raw = _raw()
    raw["vehicle"]["wheelbse_m"] = 1.0
    with pytest.raises(ValueError, match="wheelbse_m"):
        from_dict(raw)


def test_missing_key_rejected():
    raw = _raw()
    del raw["control"]["kp_speed"]
    with pytest.raises(ValueError, match="kp_speed"):
        from_dict(raw)


def test_missing_section_rejected():
    raw = _raw()
    del raw["replay"]
    with pytest.raises(ValueError, match="replay"):
        from_dict(raw)
