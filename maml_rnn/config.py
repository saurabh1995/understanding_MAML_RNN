"""Configuration: ``configs/default.yaml`` <- experiment YAML <- ``--set key.sub=value`` overrides."""

import copy
import os

import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml")


def deep_update(base, override):
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_overrides(cfg, overrides):
    """Apply ``["training.epochs=5", "model.rnn.units=16", ...]``; values are parsed as YAML."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.sub=value")
        key, raw = item.split("=", 1)
        node = cfg
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def load_config(path=None, overrides=None):
    cfg = load_yaml(DEFAULT_CONFIG)
    if path:
        cfg = deep_update(cfg, load_yaml(path))
    return apply_overrides(cfg, overrides)


def strategy_config(cfg, name=None):
    """Settings of the selected strategy (``strategies.<name>``)."""
    name = name or cfg["strategy"]
    if name not in cfg.get("strategies", {}):
        raise KeyError(f"No settings for strategy '{name}' under 'strategies' in the config")
    return cfg["strategies"][name]


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
