"""Configuration loading with recursive ``_base_`` inheritance."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config and recursively merge its optional ``_base_`` file."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    base_name = config.pop("_base_", None)
    if base_name is None:
        return config

    base_path = (config_path.parent / base_name).resolve()
    return _deep_merge(load_config(str(base_path)), config)
