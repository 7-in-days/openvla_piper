"""Strict YAML/JSON document loading shared by workspace settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Collection

import yaml


class ConfigDocumentError(ValueError):
    pass


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigDocumentError(f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def load_mapping(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            if path.suffix.lower() == ".json":
                document = json.load(stream)
            else:
                document = yaml.load(stream, Loader=_UniqueKeyLoader)
    except (OSError, json.JSONDecodeError, yaml.YAMLError, ConfigDocumentError) as exc:
        raise ConfigDocumentError(f"cannot load config {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigDocumentError(f"config root must be a mapping: {path}")
    if not all(isinstance(key, str) for key in document):
        raise ConfigDocumentError(f"config keys must be strings: {path}")
    return document


def require_mapping(parent: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigDocumentError(f"{where}.{key} must be a mapping")
    return value


def require_exact_keys(
    mapping: dict[str, Any],
    required: Collection[str],
    *,
    where: str,
    optional: Collection[str] = (),
) -> None:
    expected = set(required)
    allowed = expected | set(optional)
    actual = set(mapping)
    missing = sorted(expected - actual)
    unknown = sorted(actual - allowed)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if unknown:
            parts.append(f"unknown={unknown}")
        raise ConfigDocumentError(f"invalid keys in {where}: {' '.join(parts)}")
