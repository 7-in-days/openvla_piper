#!/usr/bin/env python3
"""Convert a LeRobotDataset v3 directory to an OpenVLA-compatible RLDS dataset."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from openvla_pipeline.cli import Option, parse_options, selected_option
from openvla_pipeline.workspace_config import DEFAULT_RLDS_CONFIG, load_rlds_config

EXPECTED_VECTOR_NAMES = [
    "joint_1.pos",
    "joint_2.pos",
    "joint_3.pos",
    "joint_4.pos",
    "joint_5.pos",
    "joint_6.pos",
    "gripper.pos",
]
IMAGE_KEYS = {
    "third_person": "observation.images.third_person",
    "wrist": "observation.images.wrist",
}


def parse_args(argv: list[str] | None = None):
    config_path = selected_option(argv, "config", Path) or DEFAULT_RLDS_CONFIG
    config = load_rlds_config(config_path)
    args, _ = parse_options(
        argv,
        (
            Option("config", converter=Path, default=config.source_path, help="RLDS YAML path"),
            Option("lerobot_root", converter=Path, default=config.source_root),
            Option("dataset_name", default=config.dataset_name),
            Option("dataset_version", default=config.dataset_version),
            Option("repo_id", default=config.repo_id),
            Option("source_revision", default=config.source_revision),
            Option("source_info_sha256", default=config.source_info_sha256),
            Option("output_root", converter=Path, default=config.output_root),
            Option("max_episodes", converter=int, default=config.max_episodes),
            Option("val_fraction", converter=float, default=config.val_fraction),
            Option("split_seed", converter=int, default=config.split_seed),
            Option("instruction", default=config.instruction),
        ),
        description="Convert PiPER LeRobotDataset v3 episodes to TFDS/RLDS.",
    )
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_info(source_root: Path) -> dict[str, Any]:
    info_path = source_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata is missing: {info_path}")
    with info_path.open("r", encoding="utf-8") as stream:
        info = json.load(stream)

    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"Expected LeRobotDataset v3.0, got {info.get('codebase_version')!r}"
        )
    if float(info.get("fps", 0)) != 20.0:
        raise ValueError(f"Expected PiPER recordings at 20 Hz, got fps={info.get('fps')!r}")
    for key in ("action", "observation.state"):
        feature = info.get("features", {}).get(key, {})
        if feature.get("dtype") != "float32" or feature.get("shape") != [7]:
            raise ValueError(f"Expected float32[7] for {key}, got {feature!r}")
        if feature.get("names") != EXPECTED_VECTOR_NAMES:
            raise ValueError(f"Unexpected {key} names: {feature.get('names')!r}")
    for key in IMAGE_KEYS.values():
        feature = info.get("features", {}).get(key, {})
        if feature.get("dtype") != "video" or feature.get("shape") != [480, 640, 3]:
            raise ValueError(f"Expected 480x640 RGB video for {key}, got {feature!r}")
    return info


def _to_uint8_hwc(value: Any, key: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.shape == (3, 480, 640):
        image = np.transpose(image, (1, 2, 0))
    if image.shape != (480, 640, 3):
        raise ValueError(f"{key} has shape {image.shape}, expected (480, 640, 3)")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError(f"{key} contains non-finite pixels")
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _to_vector(value: Any, key: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (7,):
        raise ValueError(f"{key} has shape {vector.shape}, expected (7,)")
    if not np.isfinite(vector).all():
        raise ValueError(f"{key} contains non-finite values")
    if not -1e-6 <= float(vector[6]) <= 0.085001:
        raise ValueError(f"{key} gripper value is outside [0, 0.085] m: {vector[6]}")
    return vector


def make_rlds_step(
    item: dict[str, Any], frame_offset: int, episode_length: int, instruction: str | None
) -> dict[str, Any]:
    is_last = frame_offset == episode_length - 1
    task = instruction if instruction is not None else str(item["task"])
    if not task.strip():
        raise ValueError("language instruction is empty")
    return {
        "observation": {
            "third_person": _to_uint8_hwc(item[IMAGE_KEYS["third_person"]], "third_person"),
            "wrist": _to_uint8_hwc(item[IMAGE_KEYS["wrist"]], "wrist"),
            "state": _to_vector(item["observation.state"], "observation.state"),
        },
        "action": _to_vector(item["action"], "action"),
        "discount": np.float32(1.0),
        "reward": np.float32(is_last),
        "is_first": frame_offset == 0,
        "is_last": is_last,
        "is_terminal": is_last,
        "language_instruction": task,
    }


def _select_splits(total_episodes: int, maximum: int | None, val_fraction: float, seed: int):
    if total_episodes < 1:
        raise ValueError("source dataset contains no episodes")
    if maximum is not None and maximum < 1:
        raise ValueError("--max-episodes must be positive")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("--val-fraction must be in [0, 1)")

    selected = list(range(total_episodes))
    if maximum is not None:
        selected = selected[: min(maximum, total_episodes)]
    shuffled = selected.copy()
    random.Random(seed).shuffle(shuffled)
    val_count = round(len(selected) * val_fraction)
    if val_fraction > 0 and len(selected) > 1:
        val_count = max(1, min(val_count, len(selected) - 1))
    val = set(shuffled[:val_count])
    train = [index for index in selected if index not in val]
    return train, sorted(val)


def main() -> int:
    args = parse_args()
    source_root = args.lerobot_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    source_info = _load_source_info(source_root)
    source_info_sha256 = _sha256(source_root / "meta" / "info.json")
    if args.source_info_sha256 and source_info_sha256 != args.source_info_sha256:
        raise ValueError(
            "local source metadata does not match the pinned Hugging Face revision: "
            f"expected {args.source_info_sha256}, got {source_info_sha256}"
        )
    if not args.dataset_name.replace("_", "").isalnum() or not args.dataset_name.islower():
        raise ValueError("--dataset-name must be lowercase snake_case")
    rlds_dataset_name = args.dataset_name
    rlds_dataset_version = args.dataset_version
    final_dir = output_root / rlds_dataset_name / rlds_dataset_version
    if final_dir.exists():
        raise FileExistsError(
            f"RLDS output already exists: {final_dir}; choose a new --output-root"
        )

    try:
        import tensorflow_datasets as tfds
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("RLDS tools are missing; run scripts/openvla-pipeline install-rlds") from exc

    train_episodes, val_episodes = _select_splits(
        int(source_info["total_episodes"]), args.max_episodes, args.val_fraction, args.split_seed
    )

    class PiperBridge(tfds.core.GeneratorBasedBuilder):
        name = rlds_dataset_name
        VERSION = tfds.core.Version(rlds_dataset_version)
        RELEASE_NOTES = {rlds_dataset_version: "LeRobotDataset v3 PiPER conversion."}

        def _info(self):
            return tfds.core.DatasetInfo(
                builder=self,
                description="PiPER two-block pick-and-place demonstrations in RLDS format.",
                features=tfds.features.FeaturesDict(
                    {
                        "steps": tfds.features.Dataset(
                            {
                                "observation": tfds.features.FeaturesDict(
                                    {
                                        "third_person": tfds.features.Image(
                                            shape=(480, 640, 3),
                                            dtype=np.uint8,
                                            encoding_format="jpeg",
                                        ),
                                        "wrist": tfds.features.Image(
                                            shape=(480, 640, 3),
                                            dtype=np.uint8,
                                            encoding_format="jpeg",
                                        ),
                                        "state": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                                    }
                                ),
                                "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                                "discount": tfds.features.Scalar(dtype=np.float32),
                                "reward": tfds.features.Scalar(dtype=np.float32),
                                "is_first": tfds.features.Scalar(dtype=np.bool_),
                                "is_last": tfds.features.Scalar(dtype=np.bool_),
                                "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                                "language_instruction": tfds.features.Text(),
                            }
                        ),
                        "episode_metadata": tfds.features.FeaturesDict(
                            {
                                "file_path": tfds.features.Text(),
                                "episode_index": tfds.features.Scalar(dtype=np.int64),
                            }
                        ),
                    }
                ),
                homepage="https://github.com/7-in-days/openvla_piper",
            )

        def _split_generators(self, dl_manager):
            del dl_manager
            splits = {"train": self._generate_examples(train_episodes)}
            if val_episodes:
                splits["val"] = self._generate_examples(val_episodes)
            return splits

        def _generate_examples(self, episode_indices: list[int]) -> Iterator[tuple[str, Any]]:
            dataset = LeRobotDataset(
                repo_id=args.repo_id,
                root=source_root,
                video_backend="pyav",
            )
            for episode_index in episode_indices:
                metadata = dataset.meta.episodes[int(episode_index)]
                start = int(metadata["dataset_from_index"])
                stop = int(metadata["dataset_to_index"])
                length = int(metadata["length"])
                if stop - start != length:
                    raise ValueError(
                        f"episode {episode_index} range {start}:{stop} disagrees with length {length}"
                    )
                steps = [
                    make_rlds_step(dataset[index], index - start, length, args.instruction)
                    for index in range(start, stop)
                ]
                yield str(episode_index), {
                    "steps": steps,
                    "episode_metadata": {
                        "file_path": str(source_root),
                        "episode_index": np.int64(episode_index),
                    },
                }

    output_root.mkdir(parents=True, exist_ok=True)
    builder = PiperBridge(data_dir=output_root)
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(try_download_gcs=False)
    )

    manifest = {
        "schema_version": 1,
        "dataset_name": rlds_dataset_name,
        "dataset_version": rlds_dataset_version,
        "source_format": "LeRobotDataset-v3.0",
        "source_root": str(source_root),
        "source_info_sha256": source_info_sha256,
        "repo_id": args.repo_id,
        "source_revision": args.source_revision,
        "fps": 20.0,
        "train_episode_indices": train_episodes,
        "val_episode_indices": val_episodes,
        "total_episodes": len(train_episodes) + len(val_episodes),
        "instruction_override": args.instruction,
    }
    manifest_path = final_dir / "conversion_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, manifest_path)

    loaded = tfds.builder(rlds_dataset_name, data_dir=output_root)
    print(
        "rlds_conversion=complete "
        f"path={loaded.data_dir} train_episodes={len(train_episodes)} "
        f"val_episodes={len(val_episodes)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={type(exc).__name__} detail={exc}", file=sys.stderr)
        raise
