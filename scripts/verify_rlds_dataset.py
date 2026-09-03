#!/usr/bin/env python3
"""Verify both the raw RLDS contract and OpenVLA-OFT PiPER registration."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="piper_bridge")
    parser.add_argument("--expected-episode-frames", type=int)
    parser.add_argument("--openvla-oft-repo", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    builder = tfds.builder(args.dataset_name, data_dir=data_root)
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
        if step["observation"]["third_person"].shape != (480, 640, 3):
            raise ValueError("third_person image shape mismatch")
        if step["observation"]["wrist"].shape != (480, 640, 3):
            raise ValueError("wrist image shape mismatch")
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
        pointer = Path(__file__).resolve().parents[1] / ".openvla-oft-repo"
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
        f"first_episode_frames={len(steps)} openvla_registration=PASS"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={type(exc).__name__} detail={exc}", file=sys.stderr)
        raise
