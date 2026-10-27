"""Configuration loading.

One YAML file (config/default.yaml) holds every tunable. The path comes from
the PSIRENS_CONFIG environment variable so the replay harness can point the
same pipeline at an experiment config.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


class Config:
    """Dot-accessible, read-only view over the YAML config tree."""

    def __init__(self, tree: dict[str, Any], source_path: Path | None = None):
        self._tree = tree
        self.source_path = source_path

    def __getattr__(self, key: str) -> Any:
        try:
            value = self._tree[key]
        except KeyError as e:
            raise AttributeError(f"no config key {key!r}") from e
        if isinstance(value, dict):
            return Config(value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        value = self._tree.get(key, default)
        if isinstance(value, dict):
            return Config(value)
        return value

    def as_dict(self) -> dict[str, Any]:
        return self._tree

    def digest(self) -> str:
        """Stable hash of the whole config, recorded in session metadata."""
        blob = yaml.safe_dump(self._tree, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path or os.environ.get("PSIRENS_CONFIG") or DEFAULT_CONFIG_PATH)
    with open(p) as f:
        tree = yaml.safe_load(f)
    return Config(tree, source_path=p)
