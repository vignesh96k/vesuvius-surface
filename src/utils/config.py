"""YAML configuration loading, merging, and persistence helpers."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import yaml

logger = logging.getLogger(__name__)


def _deep_update(
    base: MutableMapping[str, Any],
    overrides: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """Recursively merge ``overrides`` into ``base`` (in place)."""
    for key, value in overrides.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            _deep_update(base[key], value)  # type: ignore[arg-type]
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a single YAML file into a dictionary.

    Args:
        path: Path to a ``.yaml`` / ``.yml`` file.

    Returns:
        Parsed configuration dictionary (empty if the file is empty).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping, got {type(data)}")

    logger.debug("Loaded config: %s", path)
    return data


def load_config(
    paths: str | Path | Sequence[str | Path],
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Load and deep-merge one or more YAML configuration files.

    Later files override earlier ones. Optional ``overrides`` are applied last.

    Args:
        paths: Single path or ordered sequence of config paths.
        overrides: Optional in-memory overrides (e.g. from CLI).

    Returns:
        Merged configuration dictionary.
    """
    if isinstance(paths, (str, Path)):
        path_list: list[Path] = [Path(paths)]
    else:
        path_list = [Path(p) for p in paths]

    merged: dict[str, Any] = {}
    for path in path_list:
        _deep_update(merged, load_yaml(path))

    if overrides:
        _deep_update(merged, overrides)

    logger.info(
        "Merged %d config file(s): %s",
        len(path_list),
        ", ".join(str(p) for p in path_list),
    )
    return merged


def save_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Write a configuration dictionary to YAML.

    Args:
        config: Configuration mapping to persist.
        path: Destination YAML path.

    Returns:
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            dict(config),
            handle,
            default_flow_style=False,
            sort_keys=False,
        )
    logger.info("Saved config snapshot to %s", path)
    return path


def get_by_dotpath(config: Mapping[str, Any], dotpath: str, default: Any = None) -> Any:
    """Fetch a nested value using a dotted key path.

    Args:
        config: Nested configuration mapping.
        dotpath: Dot-separated key path (e.g. ``train.batch_size``).
        default: Value returned when the path is missing.

    Returns:
        Nested value or ``default``.
    """
    node: Any = config
    for part in dotpath.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node
