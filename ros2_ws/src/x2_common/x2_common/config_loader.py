"""Load YAML configs from the repo ``configs/`` directory.

Configs over constants (AGENTS.md §4): thresholds, ROIs, limits and gains live in
``configs/*.yaml`` and are loaded here, never hardcoded in node source.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when a config file is missing, malformed, or a key is absent."""


def _repo_configs_dir() -> Path:
    """Locate the repo ``configs/`` dir.

    Order: ``X2_CONFIG_DIR`` env override, then walk up from this file looking for a
    ``configs`` directory (works from both source tree and an installed share path).
    """
    env = os.environ.get("X2_CONFIG_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs"
        if candidate.is_dir():
            return candidate
    # Fall back to repo-root guess: <repo>/ros2_ws/src/x2_common/x2_common/this.py
    return here.parents[4] / "configs"


def config_path(name: str) -> Path:
    """Resolve a config file name (with or without ``.yaml``) to an absolute path."""
    if not name.endswith((".yaml", ".yml")):
        name = name + ".yaml"
    return _repo_configs_dir() / name


def load_config(name: str) -> dict[str, Any]:
    """Load and parse a YAML config by name. Raises :class:`ConfigError` on failure."""
    path = config_path(name)
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:  # pragma: no cover - exercised via malformed file test
        raise ConfigError(f"failed to parse {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return data


_MISSING = object()


def get(config: dict[str, Any], dotted_key: str, default: Any = _MISSING) -> Any:
    """Fetch a nested value by dotted key, e.g. ``get(cfg, "roi.x_min_m")``.

    Raises :class:`ConfigError` if the key is absent and no ``default`` is given — fail
    loudly rather than silently substituting a wrong value into safety-relevant logic.
    """
    node: Any = config
    for part in dotted_key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif default is not _MISSING:
            return default
        else:
            raise ConfigError(f"missing config key: {dotted_key!r}")
    return node
