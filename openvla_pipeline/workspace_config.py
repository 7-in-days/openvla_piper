"""Validated RLDS conversion and LoRA training YAML settings."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

from openvla_pipeline.yaml_config import (
    ConfigDocumentError,
    load_mapping,
    require_exact_keys,
    require_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RLDS_CONFIG = PROJECT_ROOT / "configs/rlds/piper_bridge.yaml"
DEFAULT_TRAINING_CONFIG = PROJECT_ROOT / "configs/training/openvla_lora.yaml"


class WorkspaceConfigError(ValueError):
    pass


def _load(path: Path, sections: set[str], kind: str) -> dict[str, Any]:
    try:
        raw = load_mapping(path)
        require_exact_keys(raw, {"schema_version", *sections}, where=kind)
    except ConfigDocumentError as exc:
        raise WorkspaceConfigError(str(exc)) from exc
    if raw.get("schema_version") != 1:
        raise WorkspaceConfigError(
            f"unsupported {kind} schema_version: {raw.get('schema_version')!r}"
        )
    return raw


def _section(
    raw: dict[str, Any], name: str, keys: set[str], kind: str
) -> dict[str, Any]:
    try:
        value = require_mapping(raw, name, kind)
        require_exact_keys(value, keys, where=f"{kind}.{name}")
    except ConfigDocumentError as exc:
        raise WorkspaceConfigError(str(exc)) from exc
    return value


def _path(value: Any, field: str, *, optional: bool = False) -> Path | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise WorkspaceConfigError(f"{field} must be a non-empty path")
    expanded = Path(os.path.expandvars(os.path.expanduser(value.strip())))
    return expanded if expanded.is_absolute() else (PROJECT_ROOT / expanded).resolve()


def _text(value: Any, field: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise WorkspaceConfigError(f"{field} must be a non-empty string")
    return value.strip()


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise WorkspaceConfigError(f"{field} must be true or false")
    return value


def _int(value: Any, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WorkspaceConfigError(f"{field} must be an integer >= {minimum}")
    return value


def _float(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkspaceConfigError(f"{field} must be a number >= {minimum}")
    result = float(value)
    if result < minimum:
        raise WorkspaceConfigError(f"{field} must be a number >= {minimum}")
    return result


@dataclass(frozen=True)
class ImageCrop:
    top: int
    left: int
    height: int
    width: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, 3)

    def as_dict(self) -> dict[str, int]:
        return {
            "top": self.top,
            "left": self.left,
            "height": self.height,
            "width": self.width,
        }


def _image_crop(value: Any, field: str) -> ImageCrop | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise WorkspaceConfigError(f"{field} must be null or a mapping")
    try:
        require_exact_keys(value, {"top", "left", "height", "width"}, where=field)
    except ConfigDocumentError as exc:
        raise WorkspaceConfigError(str(exc)) from exc
    crop = ImageCrop(
        top=_int(value["top"], f"{field}.top", minimum=0),
        left=_int(value["left"], f"{field}.left", minimum=0),
        height=_int(value["height"], f"{field}.height"),
        width=_int(value["width"], f"{field}.width"),
    )
    if crop.top + crop.height > 480 or crop.left + crop.width > 640:
        raise WorkspaceConfigError(
            f"{field} escapes the 480x640 source image: {crop.as_dict()}"
        )
    return crop


@dataclass(frozen=True)
class RldsConfig:
    source_root: Path
    repo_id: str
    source_revision: str
    source_info_sha256: str | None
    output_root: Path
    dataset_name: str
    dataset_version: str
    max_episodes: int | None
    val_fraction: float
    split_seed: int
    instruction: str | None
    image_encoding: str
    image_crops: dict[str, ImageCrop | None]
    expected_episode_frames: int | None
    openvla_oft_repo: Path | None
    source_path: Path


def load_rlds_config(path: Path | str | None = None) -> RldsConfig:
    selected = Path(path or DEFAULT_RLDS_CONFIG).expanduser().resolve()
    raw = _load(selected, {"source", "output", "split", "images", "verification"}, "rlds")
    source = _section(
        raw,
        "source",
        {"lerobot_root", "repo_id", "revision", "info_sha256", "instruction_override"},
        "rlds",
    )
    output = _section(raw, "output", {"root", "dataset_name", "dataset_version"}, "rlds")
    split = _section(raw, "split", {"max_episodes", "validation_fraction", "seed"}, "rlds")
    images = _section(raw, "images", {"encoding", "crops"}, "rlds")
    crops = _section(
        images, "crops", {"third_person", "wrist"}, "rlds.images"
    )
    verification = _section(
        raw, "verification", {"expected_episode_frames", "openvla_oft_repo"}, "rlds"
    )

    dataset_name = _text(output["dataset_name"], "rlds.output.dataset_name")
    assert dataset_name is not None
    if not re.fullmatch(r"[a-z][a-z0-9_]*", dataset_name):
        raise WorkspaceConfigError("rlds.output.dataset_name must be lowercase snake_case")
    version = _text(output["dataset_version"], "rlds.output.dataset_version")
    assert version is not None
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise WorkspaceConfigError("rlds.output.dataset_version must use MAJOR.MINOR.PATCH")
    sha = _text(source["info_sha256"], "rlds.source.info_sha256", optional=True)
    if sha is not None and not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise WorkspaceConfigError("rlds.source.info_sha256 must be 64 lowercase hex digits")
    maximum = source_max = split["max_episodes"]
    if source_max is not None:
        maximum = _int(source_max, "rlds.split.max_episodes")
    expected = verification["expected_episode_frames"]
    if expected is not None:
        expected = _int(expected, "rlds.verification.expected_episode_frames")
    val_fraction = _float(split["validation_fraction"], "rlds.split.validation_fraction")
    if val_fraction >= 1.0:
        raise WorkspaceConfigError("rlds.split.validation_fraction must be < 1")
    image_encoding = _text(images["encoding"], "rlds.images.encoding")
    if image_encoding not in {"jpeg", "png"}:
        raise WorkspaceConfigError("rlds.images.encoding must be jpeg or png")
    return RldsConfig(
        source_root=_path(source["lerobot_root"], "rlds.source.lerobot_root"),  # type: ignore[arg-type]
        repo_id=_text(source["repo_id"], "rlds.source.repo_id"),  # type: ignore[arg-type]
        source_revision=_text(source["revision"], "rlds.source.revision"),  # type: ignore[arg-type]
        source_info_sha256=sha,
        output_root=_path(output["root"], "rlds.output.root"),  # type: ignore[arg-type]
        dataset_name=dataset_name,
        dataset_version=version,
        max_episodes=maximum,
        val_fraction=val_fraction,
        split_seed=_int(split["seed"], "rlds.split.seed", minimum=0),
        instruction=_text(
            source["instruction_override"],
            "rlds.source.instruction_override",
            optional=True,
        ),
        image_encoding=image_encoding,
        image_crops={
            name: _image_crop(crops[name], f"rlds.images.crops.{name}")
            for name in ("third_person", "wrist")
        },
        expected_episode_frames=expected,
        openvla_oft_repo=_path(
            verification["openvla_oft_repo"],
            "rlds.verification.openvla_oft_repo",
            optional=True,
        ),
        source_path=selected,
    )


@dataclass(frozen=True)
class TrainingConfig:
    data_root: Path
    run_root: Path
    base_model: Path
    source_schema: Path
    robot_contract: Path
    dataset_name: str
    dataset_version: str
    control_hz: float
    action_horizon: int
    num_images_in_input: int
    use_proprio: bool
    use_film: bool
    use_l1_regression: bool
    use_diffusion: bool
    batch_size: int
    gradient_accumulation: int
    shuffle_buffer_size: int
    learning_rate: float
    lr_warmup_steps: int
    num_steps_before_decay: int
    max_steps: int
    use_val_set: bool
    val_freq: int
    val_time_limit: int
    save_freq: int
    save_latest_checkpoint_only: bool
    image_aug: bool
    lora_rank: int
    lora_alpha: int | None
    lora_dropout: float
    merge_lora_during_training: bool
    wandb_mode: str
    wandb_entity: str
    wandb_project: str
    run_id_note: str | None
    source_path: Path


def load_training_config(path: Path | str | None = None) -> TrainingConfig:
    selected = Path(path or DEFAULT_TRAINING_CONFIG).expanduser().resolve()
    raw = _load(
        selected, {"paths", "dataset", "model", "training", "lora", "tracking"}, "training"
    )
    paths = _section(
        raw, "paths", {"data_root", "run_root", "base_model", "source_schema", "robot_contract"}, "training"
    )
    dataset = _section(raw, "dataset", {"name", "version", "control_hz"}, "training")
    model = _section(
        raw,
        "model",
        {"action_horizon", "num_images_in_input", "use_proprio", "use_film", "use_l1_regression", "use_diffusion"},
        "training",
    )
    training = _section(
        raw,
        "training",
        {
            "batch_size", "gradient_accumulation", "shuffle_buffer_size", "learning_rate",
            "lr_warmup_steps", "num_steps_before_decay", "max_steps", "use_val_set",
            "val_freq", "val_time_limit", "save_freq", "save_latest_checkpoint_only", "image_aug",
        },
        "training",
    )
    lora = _section(
        raw, "lora", {"rank", "alpha", "dropout", "merge_during_training"}, "training"
    )
    tracking = _section(raw, "tracking", {"wandb_mode", "entity", "project", "run_id_note"}, "training")
    wandb_mode = _text(tracking["wandb_mode"], "training.tracking.wandb_mode")
    if wandb_mode not in {"offline", "online", "disabled"}:
        raise WorkspaceConfigError("training.tracking.wandb_mode must be offline, online, or disabled")
    alpha = lora["alpha"]
    if alpha is not None:
        alpha = _int(alpha, "training.lora.alpha")
    dropout = _float(lora["dropout"], "training.lora.dropout")
    if dropout >= 1.0:
        raise WorkspaceConfigError("training.lora.dropout must be < 1")
    note = tracking["run_id_note"]
    if note is not None and not isinstance(note, str):
        raise WorkspaceConfigError("training.tracking.run_id_note must be a string or null")
    return TrainingConfig(
        data_root=_path(paths["data_root"], "training.paths.data_root"),  # type: ignore[arg-type]
        run_root=_path(paths["run_root"], "training.paths.run_root"),  # type: ignore[arg-type]
        base_model=_path(paths["base_model"], "training.paths.base_model"),  # type: ignore[arg-type]
        source_schema=_path(paths["source_schema"], "training.paths.source_schema"),  # type: ignore[arg-type]
        robot_contract=_path(paths["robot_contract"], "training.paths.robot_contract"),  # type: ignore[arg-type]
        dataset_name=_text(dataset["name"], "training.dataset.name"),  # type: ignore[arg-type]
        dataset_version=_text(dataset["version"], "training.dataset.version"),  # type: ignore[arg-type]
        control_hz=_float(dataset["control_hz"], "training.dataset.control_hz", minimum=0.001),
        action_horizon=_int(model["action_horizon"], "training.model.action_horizon"),
        num_images_in_input=_int(model["num_images_in_input"], "training.model.num_images_in_input"),
        use_proprio=_bool(model["use_proprio"], "training.model.use_proprio"),
        use_film=_bool(model["use_film"], "training.model.use_film"),
        use_l1_regression=_bool(model["use_l1_regression"], "training.model.use_l1_regression"),
        use_diffusion=_bool(model["use_diffusion"], "training.model.use_diffusion"),
        batch_size=_int(training["batch_size"], "training.training.batch_size"),
        gradient_accumulation=_int(training["gradient_accumulation"], "training.training.gradient_accumulation"),
        shuffle_buffer_size=_int(training["shuffle_buffer_size"], "training.training.shuffle_buffer_size"),
        learning_rate=_float(training["learning_rate"], "training.training.learning_rate", minimum=1e-12),
        lr_warmup_steps=_int(training["lr_warmup_steps"], "training.training.lr_warmup_steps", minimum=0),
        num_steps_before_decay=_int(training["num_steps_before_decay"], "training.training.num_steps_before_decay"),
        max_steps=_int(training["max_steps"], "training.training.max_steps"),
        use_val_set=_bool(training["use_val_set"], "training.training.use_val_set"),
        val_freq=_int(training["val_freq"], "training.training.val_freq"),
        val_time_limit=_int(training["val_time_limit"], "training.training.val_time_limit"),
        save_freq=_int(training["save_freq"], "training.training.save_freq"),
        save_latest_checkpoint_only=_bool(training["save_latest_checkpoint_only"], "training.training.save_latest_checkpoint_only"),
        image_aug=_bool(training["image_aug"], "training.training.image_aug"),
        lora_rank=_int(lora["rank"], "training.lora.rank"),
        lora_alpha=alpha,
        lora_dropout=dropout,
        merge_lora_during_training=_bool(lora["merge_during_training"], "training.lora.merge_during_training"),
        wandb_mode=wandb_mode,
        wandb_entity=_text(tracking["entity"], "training.tracking.entity"),  # type: ignore[arg-type]
        wandb_project=_text(tracking["project"], "training.tracking.project"),  # type: ignore[arg-type]
        run_id_note=note.strip() if isinstance(note, str) and note.strip() else None,
        source_path=selected,
    )
