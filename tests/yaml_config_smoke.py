#!/usr/bin/env python3
"""Network-free checks for strict YAML settings and the internal CLI parser."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from openvla_pipeline.cli import Option, parse_options
from openvla_pipeline.config import RuntimeConfigError, load_runtime_config
from openvla_pipeline.workspace_config import load_rlds_config, load_training_config
from scripts.train_openvla_lora import _materialize_robot_contract


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
assert rlds.image_encoding == "jpeg"
assert rlds.image_crops == {"third_person": None, "wrist": None}
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
    root = Path(temporary_directory)
    duplicate = Path(temporary_directory) / "duplicate.yaml"
    duplicate.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
    _expect(RuntimeConfigError, lambda: load_runtime_config(duplicate))

    custom_rlds = root / "rlds.yaml"
    custom_rlds.write_text(
        (PROJECT_ROOT / "configs/rlds/piper_bridge.yaml")
        .read_text(encoding="utf-8")
        .replace("encoding: jpeg", "encoding: png")
        .replace(
            "third_person: null",
            "third_person: {top: 60, left: 80, height: 360, width: 480}",
        ),
        encoding="utf-8",
    )
    custom = load_rlds_config(custom_rlds)
    assert custom.image_encoding == "png"
    assert custom.image_crops["third_person"].shape == (360, 480, 3)

    invalid_rlds = root / "invalid-rlds.yaml"
    invalid_rlds.write_text(
        custom_rlds.read_text(encoding="utf-8").replace("height: 360", "height: 421"),
        encoding="utf-8",
    )
    _expect(ValueError, lambda: load_rlds_config(invalid_rlds))

    dataset_dir = root / "data" / training.dataset_name / training.dataset_version
    dataset_dir.mkdir(parents=True)
    image_preprocessing = {
        "source_shape": [480, 640, 3],
        "encoding": "png",
        "crops": {
            "third_person": {"top": 60, "left": 80, "height": 360, "width": 480},
            "wrist": None,
        },
        "output_shapes": {
            "third_person": [360, 480, 3],
            "wrist": [480, 640, 3],
        },
    }
    (dataset_dir / "conversion_manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": training.dataset_name,
                "dataset_version": training.dataset_version,
                "image_preprocessing": image_preprocessing,
            }
        ),
        encoding="utf-8",
    )
    resolved_contract_path = _materialize_robot_contract(
        replace(training, data_root=root / "data", run_root=root / "run")
    )
    resolved_contract = json.loads(resolved_contract_path.read_text(encoding="utf-8"))
    assert resolved_contract["robot_type"] == "piper_bridge"
    assert resolved_contract["image_preprocessing"] == image_preprocessing

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
print("training_image_preprocessing_contract=manifest_to_checkpoint")
print("python_cli_parser=internal argparse_imports=0")
