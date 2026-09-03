#!/usr/bin/env python3
"""Verify both the raw RLDS contract and OpenVLA-OFT PiPER registration."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from openvla_pipeline.cli import Option, parse_options, selected_option
from openvla_pipeline.workspace_config import DEFAULT_RLDS_CONFIG, load_rlds_config


def parse_args(argv: list[str] | None = None):
    config_path = selected_option(argv, "config", Path) or DEFAULT_RLDS_CONFIG
    config = load_rlds_config(config_path)
    args, _ = parse_options(
        argv,
        (
            Option("config", converter=Path, default=config.source_path, help="RLDS YAML path"),
            Option("data_root", converter=Path, default=config.output_root),
            Option("dataset_name", default=config.dataset_name),
            Option(
                "expected_episode_frames",
                converter=int,
                default=config.expected_episode_frames,
            ),
            Option(
                "image_encoding",
                default=config.image_encoding,
                choices=("jpeg", "png"),
            ),
            Option("openvla_oft_repo", converter=Path, default=config.openvla_oft_repo),
        ),
        description="Verify raw RLDS and OpenVLA-OFT PiPER registration.",
    )
    args.image_crops = config.image_crops
    return args


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    builder = tfds.builder(args.dataset_name, data_dir=data_root)
    manifest_path = Path(builder.data_dir) / "conversion_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"conversion manifest is missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    expected_preprocessing = {
        "source_shape": [480, 640, 3],
        "encoding": args.image_encoding,
        "crops": {
            name: crop.as_dict() if crop is not None else None
            for name, crop in args.image_crops.items()
        },
        "output_shapes": {
            name: list(crop.shape if crop is not None else (480, 640, 3))
            for name, crop in args.image_crops.items()
        },
    }
    if manifest.get("image_preprocessing") != expected_preprocessing:
        raise ValueError(
            "RLDS image preprocessing does not match config: "
            f"actual={manifest.get('image_preprocessing')!r} "
            f"expected={expected_preprocessing!r}"
        )
    observation_features = builder.info.features["steps"].feature["observation"]
    for camera_name, expected_shape in expected_preprocessing["output_shapes"].items():
        feature = observation_features[camera_name]
        if feature.encoding_format != args.image_encoding:
            raise ValueError(
                f"{camera_name} encoding mismatch: {feature.encoding_format}"
            )
        if tuple(feature.shape) != tuple(expected_shape):
            raise ValueError(f"{camera_name} TFDS feature shape mismatch: {feature.shape}")
    split_names = set(builder.info.splits)
    if "train" not in split_names:
        raise ValueError(f"missing train split: {sorted(split_names)}")

    dataset = builder.as_dataset(split="train", shuffle_files=False)
    episode = next(iter(dataset.take(1)), None)
    if episode is None:
        raise ValueError("train split contains no episodes")
    steps = list(episode["steps"].as_numpy_iterator())
    if not steps:
        raise ValueError("first train episode contains no steps")
    if args.expected_episode_frames is not None and len(steps) != args.expected_episode_frames:
        raise ValueError(
            f"first train episode has {len(steps)} frames, expected {args.expected_episode_frames}"
        )

    for index in {0, len(steps) - 1}:
        step = steps[index]
        for camera_name, expected_shape in expected_preprocessing["output_shapes"].items():
            if step["observation"][camera_name].shape != tuple(expected_shape):
                raise ValueError(f"{camera_name} image shape mismatch")
        for key, vector in (
            ("state", step["observation"]["state"]),
            ("action", step["action"]),
        ):
            vector = np.asarray(vector)
            if vector.shape != (7,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
                raise ValueError(f"invalid {key}: shape={vector.shape} dtype={vector.dtype}")
        if not step["language_instruction"].decode("utf-8").strip():
            raise ValueError("empty language instruction")

    if not bool(steps[0]["is_first"]) or bool(steps[0]["is_last"]):
        raise ValueError("invalid first-step RLDS flags")
    if not bool(steps[-1]["is_last"]) or not bool(steps[-1]["is_terminal"]):
        raise ValueError("invalid final-step RLDS flags")

    oft_repo = args.openvla_oft_repo
    if oft_repo is None:
        pointer = PROJECT_ROOT / ".openvla-oft-repo"
        if not pointer.is_file():
            raise FileNotFoundError(f"OpenVLA-OFT pointer is missing: {pointer}")
        oft_repo = Path(pointer.read_text(encoding="utf-8").strip())
    oft_repo = oft_repo.expanduser().resolve()
    sys.path.insert(0, str(oft_repo))
    os.environ.setdefault("ROBOT_PLATFORM", "PIPER")
    os.environ.setdefault("PIPER_ACTION_CHUNK", "50")

    from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs

    kwargs = make_oxe_dataset_kwargs(
        args.dataset_name,
        data_root,
        load_camera_views=("primary", "wrist"),
        load_depth=False,
        load_proprio=True,
        load_language=True,
    )
    if kwargs["image_obs_keys"] != {
        "primary": "third_person",
        "wrist": "wrist",
    }:
        raise ValueError(f"unexpected OpenVLA camera mapping: {kwargs['image_obs_keys']}")
    if kwargs["state_obs_keys"] != ["state"] or kwargs["language_key"] != "language_instruction":
        raise ValueError("OpenVLA PiPER state/language registration mismatch")

    print(
        "rlds_contract=PASS "
        f"dataset={args.dataset_name} train_episodes={builder.info.splits['train'].num_examples} "
        f"first_episode_frames={len(steps)} image_encoding={args.image_encoding} "
        f"image_shapes={expected_preprocessing['output_shapes']} openvla_registration=PASS"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={type(exc).__name__} detail={exc}", file=sys.stderr)
        raise
