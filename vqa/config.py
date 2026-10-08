"""YAML configuration: a default file, optional named run overrides, and key=value overrides."""
import copy
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def merge(base, extra):
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge(base[k], v)
        else:
            base[k] = v
    return base


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path="configs/default.yaml", run=None, overrides=()):
    """Default config, then the file at `path`, then the named run, then `a.b=value` overrides."""
    default = os.path.join(ROOT, "configs", "default.yaml")
    path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    cfg = load_yaml(default)
    if os.path.normpath(path) != os.path.normpath(default):
        merge(cfg, load_yaml(path))
    if run:
        runs = merge(load_yaml(os.path.join(ROOT, "configs", "runs.yaml")), cfg.pop("runs", None))
        merge(cfg, copy.deepcopy(runs[run]))
        cfg["run"] = run
    cfg.pop("runs", None)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    for k, v in cfg["paths"].items():  # paths in the config are relative to the project root
        cfg["paths"][k] = v if os.path.isabs(v) else os.path.join(ROOT, v)
    return cfg
