from __future__ import annotations

from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Raised when a config file is malformed or uses unsupported options."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of silently keeping the last one."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ConfigError(
                f"Duplicate key {key!r} in {node.start_mark.name}, line {key_node.start_mark.line + 1}"
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def load_yaml_strict(path: str | Path) -> dict:
    """Load YAML, failing on duplicate keys."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_UniqueKeyLoader) or {}
