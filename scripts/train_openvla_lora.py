#!/usr/bin/env python3
"""Build and execute the official OpenVLA-OFT LoRA command from YAML."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from openvla_pipeline.cli import Option, parse_options, selected_option
from openvla_pipeline.workspace_config import (
    DEFAULT_TRAINING_CONFIG,
    TrainingConfig,
    load_training_config,
)


def _pointer(name: str) -> str | None:
    path = PROJECT_ROOT / name
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def _bool_text(value: bool) -> str:
    return "True" if value else "False"


def parse_settings(argv: list[str] | None = None) -> tuple[TrainingConfig, bool]:
    config_path = selected_option(argv, "config", Path) or DEFAULT_TRAINING_CONFIG
    config = load_training_config(config_path)
    args, _ = parse_options(
        argv,
        (
            Option("config", converter=Path, default=config.source_path, help="training YAML path"),
            Option("data_root", converter=Path, default=config.data_root),
            Option("run_root", converter=Path, default=config.run_root),
            Option("base_model", converter=Path, default=config.base_model),
            Option("source_schema", converter=Path, default=config.source_schema),
            Option("dataset_name", default=config.dataset_name),
            Option("control_hz", converter=float, default=config.control_hz),
            Option("action_horizon", converter=int, default=config.action_horizon),
            Option("batch_size", converter=int, default=config.batch_size),
            Option("gradient_accumulation", converter=int, default=config.gradient_accumulation),
            Option("max_steps", converter=int, default=config.max_steps),
            Option("save_freq", converter=int, default=config.save_freq),
            Option("learning_rate", converter=float, default=config.learning_rate),
            Option("lora_rank", converter=int, default=config.lora_rank),
            Option(
                "wandb_mode",
                default=config.wandb_mode,
                choices=("offline", "online", "disabled"),
            ),
            Option("dry_run", switch=True, default=False, help="print command only"),
        ),
        description="Run official OpenVLA-OFT synchronous LoRA fine-tuning.",
    )
    for name in (
        "action_horizon", "batch_size", "gradient_accumulation", "max_steps",
        "save_freq", "lora_rank",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.control_hz <= 0 or args.learning_rate <= 0:
        raise ValueError("--control-hz and --learning-rate must be positive")
    return (
        replace(
            config,
            data_root=args.data_root.expanduser().resolve(),
            run_root=args.run_root.expanduser().resolve(),
            base_model=args.base_model.expanduser().resolve(),
            source_schema=args.source_schema.expanduser().resolve(),
            dataset_name=args.dataset_name,
            control_hz=args.control_hz,
            action_horizon=args.action_horizon,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            max_steps=args.max_steps,
            save_freq=args.save_freq,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            wandb_mode=args.wandb_mode,
        ),
        args.dry_run,
    )


def _require_file(path: Path, error: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{error} path={path}")


def _materialize_robot_contract(config: TrainingConfig) -> Path:
    """Bind the immutable RLDS image transform to each newly trained checkpoint."""
    _require_file(config.robot_contract, "robot_contract_missing")
    manifest_path = (
        config.data_root
        / config.dataset_name
        / config.dataset_version
        / "conversion_manifest.json"
    )
    _require_file(manifest_path, "rlds_conversion_manifest_missing")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("dataset_name") != config.dataset_name:
        raise ValueError("RLDS manifest dataset_name does not match training config")
    if manifest.get("dataset_version") != config.dataset_version:
        raise ValueError("RLDS manifest dataset_version does not match training config")
    image_preprocessing = manifest.get("image_preprocessing")
    if not isinstance(image_preprocessing, dict):
        raise ValueError(
            "RLDS manifest has no image_preprocessing contract; reconvert the dataset"
        )

    with config.robot_contract.open("r", encoding="utf-8") as stream:
        robot_contract = json.load(stream)
    camera_names = robot_contract.get("camera_names")
    crops = image_preprocessing.get("crops")
    output_shapes = image_preprocessing.get("output_shapes")
    if (
        not isinstance(camera_names, list)
        or not isinstance(crops, dict)
        or not isinstance(output_shapes, dict)
    ):
        raise ValueError("invalid robot or RLDS image preprocessing contract")
    if set(crops) != set(camera_names) or set(output_shapes) != set(camera_names):
        raise ValueError("RLDS image preprocessing camera names do not match robot contract")
    robot_contract["image_preprocessing"] = image_preprocessing
    from openvla_pipeline.openvla_policy import PiperOpenVLAPolicy

    PiperOpenVLAPolicy._parse_image_preprocessing(robot_contract)

    canonical = json.dumps(robot_contract, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    contract_dir = config.run_root / ".contracts"
    contract_dir.mkdir(parents=True, exist_ok=True)
    output_path = contract_dir / f"{config.dataset_name}-{config.dataset_version}-{digest}.json"
    if not output_path.is_file():
        temporary = output_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(robot_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output_path)
    return output_path


def build_command(config: TrainingConfig) -> tuple[list[str], dict[str, str]]:
    runtime_prefix = os.environ.get("OPENVLA_CONDA_PREFIX") or _pointer(".install-prefix")
    oft_text = os.environ.get("OPENVLA_OFT_REPO") or _pointer(".openvla-oft-repo")
    if not runtime_prefix:
        raise RuntimeError("openvla_runtime_missing hint=run_scripts/install_openvla.sh")
    if not oft_text:
        raise RuntimeError("openvla_oft_source_missing hint=run_scripts/install_openvla.sh")
    torchrun = Path(runtime_prefix).expanduser().resolve() / "bin/torchrun"
    oft_repo = Path(oft_text).expanduser().resolve()
    finetune = oft_repo / "vla-scripts/finetune.py"
    _require_file(torchrun, "openvla_runtime_missing")
    _require_file(finetune, "openvla_oft_source_missing")
    _require_file(
        config.data_root / config.dataset_name / config.dataset_version / "dataset_info.json",
        "rlds_dataset_missing hint=run_scripts/openvla-pipeline_convert-rlds",
    )
    _require_file(config.base_model / "config.json", "base_model_missing")
    _require_file(config.source_schema, "source_schema_missing")
    _require_file(config.robot_contract, "robot_contract_missing")

    run_note = config.run_id_note or (
        f"piper--{config.action_horizon}_actions--{config.control_hz:g}hz--"
        f"{config.num_images_in_input}_images--proprio--lora"
    )
    command = [
        str(torchrun), "--standalone", "--nnodes", "1", "--nproc-per-node", "1",
        str(finetune),
        "--vla_path", str(config.base_model),
        "--data_root_dir", str(config.data_root),
        "--dataset_name", config.dataset_name,
        "--control_hz", str(config.control_hz),
        "--robot_contract_path", str(config.robot_contract),
        "--source_schema_path", str(config.source_schema),
        "--run_root_dir", str(config.run_root),
        "--shuffle_buffer_size", str(config.shuffle_buffer_size),
        "--use_l1_regression", _bool_text(config.use_l1_regression),
        "--use_diffusion", _bool_text(config.use_diffusion),
        "--use_film", _bool_text(config.use_film),
        "--num_images_in_input", str(config.num_images_in_input),
        "--use_proprio", _bool_text(config.use_proprio),
        "--batch_size", str(config.batch_size),
        "--learning_rate", str(config.learning_rate),
        "--lr_warmup_steps", str(config.lr_warmup_steps),
        "--num_steps_before_decay", str(config.num_steps_before_decay),
        "--grad_accumulation_steps", str(config.gradient_accumulation),
        "--max_steps", str(config.max_steps),
        "--use_val_set", _bool_text(config.use_val_set),
        "--val_freq", str(config.val_freq),
        "--val_time_limit", str(config.val_time_limit),
        "--save_freq", str(config.save_freq),
        "--save_latest_checkpoint_only", _bool_text(config.save_latest_checkpoint_only),
        "--image_aug", _bool_text(config.image_aug),
        "--lora_rank", str(config.lora_rank),
        "--lora_dropout", str(config.lora_dropout),
        "--merge_lora_during_training", _bool_text(config.merge_lora_during_training),
        "--wandb_entity", config.wandb_entity,
        "--wandb_project", config.wandb_project,
        "--run_id_note", run_note,
    ]
    if config.lora_alpha is not None:
        command.extend(("--lora_alpha", str(config.lora_alpha)))

    cache_root = PROJECT_ROOT / ".runtime/huggingface-training"
    environment = os.environ.copy()
    environment.update(
        {
            "ROBOT_PLATFORM": "PIPER",
            "PIPER_ACTION_CHUNK": str(config.action_horizon),
            "WANDB_MODE": config.wandb_mode,
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "PYTHONNOUSERSITE": "1",
            "HF_HOME": str(cache_root),
            "HF_DATASETS_CACHE": str(cache_root / "datasets"),
        }
    )
    return command, environment


def main() -> int:
    config, dry_run = parse_settings()
    config.run_root.mkdir(parents=True, exist_ok=True)
    config = replace(config, robot_contract=_materialize_robot_contract(config))
    command, environment = build_command(config)
    Path(environment["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    print(
        "training_contract=ready "
        f"config={config.source_path} dataset={config.dataset_name} "
        f"action_horizon={config.action_horizon} control_hz={config.control_hz:g} "
        f"chunk_seconds={config.action_horizon / config.control_hz:.2f} "
        f"effective_batch={config.batch_size * config.gradient_accumulation}",
        flush=True,
    )
    if dry_run:
        assignments = [
            f"{name}={environment[name]}"
            for name in (
                "ROBOT_PLATFORM", "PIPER_ACTION_CHUNK", "WANDB_MODE",
                "TF_FORCE_GPU_ALLOW_GROWTH", "PYTHONNOUSERSITE", "HF_HOME",
                "HF_DATASETS_CACHE",
            )
        ]
        print(shlex.join(["env", *assignments, *command]))
        return 0
    os.execvpe(command[0], command, environment)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error={type(exc).__name__} detail={exc}", file=sys.stderr)
        raise SystemExit(1) from exc
