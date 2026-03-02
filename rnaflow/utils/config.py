"""YAML configuration loader."""

from pathlib import Path
from types import SimpleNamespace

import yaml


def load_config(path: str | Path) -> SimpleNamespace:
    """Load a YAML config file and return it as a SimpleNamespace for attribute access."""
    with open(path) as f:
        d = yaml.safe_load(f)
    return _dict_to_namespace(d)


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert nested dicts to SimpleNamespace."""
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = _dict_to_namespace(value)
    return SimpleNamespace(**d)
