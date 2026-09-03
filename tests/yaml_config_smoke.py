#!/usr/bin/env python3
"""Network-free checks for strict YAML settings and the internal CLI parser."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from openvla_pipeline.cli import Option, parse_options
from openvla_pipeline.config import RuntimeConfigError, load_runtime_config
from openvla_pipeline.workspace_config import load_rlds_config, load_training_config


def _expect(error_type: type[Exception], callback) -> None:
    try:
        callback()
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


runtime = load_runtime_config()
rlds = load_rlds_config()
training = load_training_config()
assert runtime.source_path.name == "openvla_piper.yaml"
assert runtime.client.max_actions == 500
assert rlds.dataset_name == training.dataset_name == "piper_bridge"
assert rlds.expected_episode_frames == 700
assert training.action_horizon == 50 and training.control_hz == 20.0

values, forwarded = parse_options(
    ["--no-enabled", "--count=7"],
    (
        Option("enabled", boolean=True, default=True),
        Option("count", converter=int, default=1),
    ),
    description="test",
)
assert not values.enabled and values.count == 7 and forwarded == []

with tempfile.TemporaryDirectory() as temporary_directory:
    duplicate = Path(temporary_directory) / "duplicate.yaml"
    duplicate.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
    _expect(RuntimeConfigError, lambda: load_runtime_config(duplicate))

for source_root in (PROJECT_ROOT / "openvla_pipeline", PROJECT_ROOT / "scripts"):
    for source_path in source_root.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name != "argparse" for alias in node.names), source_path
            if isinstance(node, ast.ImportFrom):
                assert node.module != "argparse", source_path

print("yaml_config=PASS strict_keys=True duplicate_keys_rejected=True")
print("workspace_defaults=runtime,rlds,training")
print("python_cli_parser=internal argparse_imports=0")
