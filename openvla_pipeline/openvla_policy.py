"""Load an OpenVLA-OFT checkpoint and predict Piper action chunks."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from openvla_pipeline.model_io import (
    ACTION_CHUNK,
    ACTION_KEYS,
    PiperActionSafetyGuard,
    ContractError,
)
from openvla_pipeline.checkpoint_source import resolve_source


class PiperOpenVLAPolicy:
    """Loads an atomic OpenVLA-OFT checkpoint and serves one Hx7 action chunk per call."""

    def __init__(
        self,
        checkpoint: str | Path,
        base_model: str | Path | None = None,
        openvla_oft_repo: Path | None = None,
        max_arm_step_delta_rad: float = 1.5,
    ) -> None:
        checkpoint_source = resolve_source(checkpoint)
        self.checkpoint_source = checkpoint_source.requested
        self.checkpoint_source_kind = checkpoint_source.kind
        self.checkpoint = checkpoint_source.local_path
        self._metadata = self._load_and_validate_metadata()
        contract = self._metadata["training_contract"]
        self.normalization = str(contract["normalization"])
        self.action_chunk = int(contract["num_actions_chunk"])
        self.action_dim = int(contract["action_dim"])
        self.action_shape = (self.action_chunk, self.action_dim)
        self.num_images_in_input = int(contract["num_images_in_input"])
        self.use_film = bool(contract["use_film"])
        self.image_augmentation = bool(contract["image_aug"])
        self.robot_contract = contract["robot_contract"]
        self.image_keys = tuple(self.robot_contract["camera_names"])
        self.image_preprocessing = self._parse_image_preprocessing(self.robot_contract)
        self.unnorm_key = str(self.robot_contract["robot_type"])
        self._configure_oft_environment()
        recorded_base = str(self._metadata["base_vla_path"])
        base_source = resolve_source(base_model or recorded_base)
        self.base_model_source = base_source.requested
        self.base_model_source_kind = base_source.kind
        self.base_model = base_source.local_path
        self.openvla_oft_repo = self._resolve_openvla_oft_repo(openvla_oft_repo)
        self._validate_files()
        self.action_safety = PiperActionSafetyGuard.from_statistics(
            self.checkpoint / "dataset_statistics.json",
            self.normalization,
            self.action_shape,
            self.unnorm_key,
            max_arm_step_delta_rad,
        )
        self._lock = threading.Lock()
        self._log_load_event(
            "model_load_start",
            checkpoint=str(self.checkpoint),
            base_model=str(self.base_model),
        )
        self._load_model()

    @staticmethod
    def _parse_image_preprocessing(robot_contract: dict[str, Any]) -> dict[str, Any] | None:
        preprocessing = robot_contract.get("image_preprocessing")
        if preprocessing is None:
            return None
        if not isinstance(preprocessing, dict):
            raise ContractError("image_preprocessing must be an object")
        if set(preprocessing) != {"source_shape", "encoding", "crops", "output_shapes"}:
            raise ContractError("image_preprocessing fields are invalid")
        source_shape = preprocessing["source_shape"]
        if source_shape != [480, 640, 3]:
            raise ContractError(
                f"unsupported image_preprocessing source_shape: {source_shape!r}"
            )
        if preprocessing["encoding"] not in {"jpeg", "png"}:
            raise ContractError("image_preprocessing encoding must be jpeg or png")
        camera_names = tuple(robot_contract["camera_names"])
        crops = preprocessing["crops"]
        output_shapes = preprocessing["output_shapes"]
        if not isinstance(crops, dict) or set(crops) != set(camera_names):
            raise ContractError("image_preprocessing crops do not match camera_names")
        if not isinstance(output_shapes, dict) or set(output_shapes) != set(camera_names):
            raise ContractError("image_preprocessing output_shapes do not match camera_names")
        for camera_name in camera_names:
            crop = crops[camera_name]
            if crop is None:
                expected_shape = source_shape
            else:
                if not isinstance(crop, dict) or set(crop) != {"top", "left", "height", "width"}:
                    raise ContractError(f"invalid {camera_name} image crop")
                values = tuple(crop[key] for key in ("top", "left", "height", "width"))
                if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                    raise ContractError(f"non-integer {camera_name} image crop")
                top, left, height, width = values
                if top < 0 or left < 0 or height < 1 or width < 1:
                    raise ContractError(f"invalid {camera_name} image crop bounds")
                if top + height > source_shape[0] or left + width > source_shape[1]:
                    raise ContractError(f"{camera_name} image crop escapes source image")
                expected_shape = [height, width, 3]
            if output_shapes[camera_name] != expected_shape:
                raise ContractError(f"{camera_name} image output shape disagrees with crop")
        return preprocessing

    def _prepare_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(observation)
        prepared["state"] = np.asarray(observation["state"], dtype=np.float32).copy()
        if self.image_preprocessing is None:
            return prepared
        transport_names = {"third_person": "full_image", "wrist": "wrist_image"}
        source_shape = tuple(self.image_preprocessing["source_shape"])
        for camera_name in self.image_keys:
            transport_name = transport_names[camera_name]
            image = np.asarray(observation[transport_name])
            if image.shape != source_shape or image.dtype != np.uint8:
                raise ContractError(
                    f"{camera_name} must be uint8 shape {source_shape} before trained crop, "
                    f"got shape={image.shape} dtype={image.dtype}"
                )
            crop = self.image_preprocessing["crops"][camera_name]
            if crop is not None:
                top, left = crop["top"], crop["left"]
                image = image[top:top + crop["height"], left:left + crop["width"], :]
            prepared[transport_name] = np.ascontiguousarray(image)
        return prepared

    @staticmethod
    def _log_load_event(event: str, **fields: Any) -> None:
        print(json.dumps({"event": event, **fields}, separators=(",", ":")), flush=True)

    def _load_and_validate_metadata(self) -> dict[str, Any]:
        metadata_path = self.checkpoint / "checkpoint_metadata.json"
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        contract = metadata.get("training_contract", {})
        action_dim = contract.get("action_dim")
        action_chunk = contract.get("num_actions_chunk")
        num_images = contract.get("num_images_in_input")
        expected = {
            "proprio_dim": action_dim,
            "use_l1_regression": True,
            "use_proprio": True,
            "use_diffusion": False,
            "use_film": False,
        }
        mismatches = {
            key: (contract.get(key), value)
            for key, value in expected.items()
            if contract.get(key) != value
        }
        if mismatches:
            raise ContractError(f"checkpoint training contract mismatch: {mismatches}")
        if not isinstance(action_dim, int) or action_dim <= 0:
            raise ContractError(f"checkpoint action_dim must be positive, got {action_dim!r}")
        if not isinstance(action_chunk, int) or action_chunk <= 0:
            raise ContractError(
                f"checkpoint num_actions_chunk must be positive, got {action_chunk!r}"
            )
        if not isinstance(num_images, int) or num_images <= 0:
            raise ContractError(
                f"checkpoint num_images_in_input must be positive, got {num_images!r}"
            )
        if ACTION_CHUNK is not None and ACTION_CHUNK != action_chunk:
            raise ContractError(
                "PIPER_ACTION_CHUNK conflicts with checkpoint metadata: "
                f"environment={ACTION_CHUNK} checkpoint={action_chunk}"
            )
        robot_contract = contract.get("robot_contract")
        if not isinstance(robot_contract, dict):
            raise ContractError("checkpoint is missing training_contract.robot_contract")
        expected_units = ("radian",) * 6 + ("meter",)
        expected_robot_contract = {
            "robot_type": "piper_bridge",
            "action_encoding": "absolute_joint_position",
            "action_names": ACTION_KEYS,
            "state_names": ACTION_KEYS,
            "action_units": expected_units,
            "state_units": expected_units,
            "camera_names": ("third_person", "wrist"),
        }
        robot_mismatches = {
            key: {
                "actual": tuple(robot_contract.get(key, ()))
                if isinstance(expected_value, tuple)
                else robot_contract.get(key),
                "expected": expected_value,
            }
            for key, expected_value in expected_robot_contract.items()
            if (
                tuple(robot_contract.get(key, ()))
                if isinstance(expected_value, tuple)
                else robot_contract.get(key)
            )
            != expected_value
        }
        if robot_mismatches:
            raise ContractError(
                "checkpoint is not compatible with the Piper semantic contract: "
                f"{robot_mismatches}"
            )
        camera_names = tuple(robot_contract["camera_names"])
        if len(camera_names) != num_images:
            raise ContractError(
                "checkpoint camera count mismatch: "
                f"camera_names={camera_names} num_images_in_input={num_images}"
            )
        if not isinstance(contract.get("image_aug"), bool):
            raise ContractError(
                "checkpoint training contract image_aug must be a boolean, "
                f"got {contract.get('image_aug')!r}"
            )
        if contract.get("normalization") not in ("bounds", "bounds_q99"):
            raise ContractError(
                "checkpoint training contract has unsupported normalization: "
                f"{contract.get('normalization')!r}"
            )
        return metadata

    def _configure_oft_environment(self) -> None:
        """Bind checkpoint-owned Piper constants before importing OpenVLA-OFT."""
        configured_chunk = os.environ.get("PIPER_ACTION_CHUNK")
        if configured_chunk is not None:
            try:
                parsed_chunk = int(configured_chunk)
            except ValueError as exc:
                raise ContractError(
                    "PIPER_ACTION_CHUNK must be a positive integer, "
                    f"got {configured_chunk!r}"
                ) from exc
            if parsed_chunk <= 0:
                raise ContractError(
                    "PIPER_ACTION_CHUNK must be a positive integer, "
                    f"got {configured_chunk!r}"
                )
            if parsed_chunk != self.action_chunk:
                raise ContractError(
                    "PIPER_ACTION_CHUNK conflicts with checkpoint metadata: "
                    f"environment={parsed_chunk} checkpoint={self.action_chunk}"
                )

        configured_platform = os.environ.get("ROBOT_PLATFORM")
        if configured_platform is not None and configured_platform.strip().upper() != "PIPER":
            raise ContractError(
                "ROBOT_PLATFORM conflicts with the Piper checkpoint: "
                f"environment={configured_platform!r} checkpoint='PIPER'"
            )

        os.environ["PIPER_ACTION_CHUNK"] = str(self.action_chunk)
        os.environ["ROBOT_PLATFORM"] = "PIPER"

    def _validate_oft_constants(self, constants_module: Any) -> None:
        """Fail before weight loading when the selected OFT checkout is incompatible."""
        module_file = Path(constants_module.__file__).resolve()
        try:
            module_file.relative_to(self.openvla_oft_repo)
        except ValueError as exc:
            raise ContractError(
                "OpenVLA-OFT constants were imported from an unexpected checkout: "
                f"module={module_file} expected_repo={self.openvla_oft_repo}"
            ) from exc

        normalization = getattr(
            constants_module.ACTION_PROPRIO_NORMALIZATION_TYPE,
            "value",
            constants_module.ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        actual = {
            "robot_platform": str(constants_module.ROBOT_PLATFORM),
            "action_chunk": int(constants_module.NUM_ACTIONS_CHUNK),
            "action_dim": int(constants_module.ACTION_DIM),
            "proprio_dim": int(constants_module.PROPRIO_DIM),
            "normalization": str(normalization),
        }
        expected = {
            "robot_platform": "PIPER",
            "action_chunk": self.action_chunk,
            "action_dim": self.action_dim,
            "proprio_dim": self.action_dim,
            "normalization": self.normalization,
        }
        mismatches = {
            key: {"actual": actual[key], "expected": expected[key]}
            for key in expected
            if actual[key] != expected[key]
        }
        if mismatches:
            raise ContractError(
                "OpenVLA-OFT constants do not match checkpoint metadata: "
                f"{mismatches}"
            )
        self._resolved_oft_contract = {
            "constants_module": str(module_file),
            "repository": str(self.openvla_oft_repo),
            **actual,
        }

    @staticmethod
    def _resolve_openvla_oft_repo(explicit: Path | None) -> Path:
        environment = os.environ.get("OPENVLA_OFT_REPO")
        candidates = [
            explicit,
            Path(environment) if environment else None,
            Path(__file__).resolve().parents[2] / "repositories/openvla-oft",
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            resolved = candidate.expanduser().resolve()
            if (resolved / "experiments/robot/openvla_utils.py").is_file():
                return resolved
        raise FileNotFoundError(
            "OpenVLA-OFT source was not found; set --openvla-oft-repo or OPENVLA_OFT_REPO"
        )

    def _validate_files(self) -> None:
        required = (
            self.base_model / "config.json",
            self.checkpoint / "lora_adapter" / "adapter_config.json",
            self.checkpoint / "lora_adapter" / "adapter_model.safetensors",
            self.checkpoint / "action_head.pt",
            self.checkpoint / "proprio_projector.pt",
            self.checkpoint / "dataset_statistics.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing deployment checkpoint files: {missing}")

    def _load_model(self) -> None:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

        if str(self.openvla_oft_repo) not in sys.path:
            sys.path.insert(0, str(self.openvla_oft_repo))

        import tensorflow as tf

        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as exc:
            raise RuntimeError("TensorFlow initialized CUDA before deployment policy setup") from exc

        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

        from experiments.robot.openvla_utils import get_vla_action
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        from prismatic.models.action_heads import L1RegressionActionHead
        from prismatic.models.projectors import ProprioProjector
        from prismatic.vla import constants as oft_constants

        self._validate_oft_constants(oft_constants)

        if not torch.cuda.is_available():
            raise RuntimeError("OpenVLA deployment requires a CUDA GPU")

        AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True)

        device = torch.device("cuda:0")
        processor = AutoProcessor.from_pretrained(
            self.base_model,
            trust_remote_code=False,
            local_files_only=True,
        )
        base_vla = AutoModelForVision2Seq.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            local_files_only=True,
        )
        self._log_load_event("base_model_loaded", base_model=str(self.base_model))
        lora_adapter = self.checkpoint / "lora_adapter"
        peft_vla = PeftModel.from_pretrained(
            base_vla,
            lora_adapter,
            is_trainable=False,
            local_files_only=True,
        )
        self._log_load_event("lora_adapter_attached", lora_adapter=str(lora_adapter))
        # Deployment uses one immutable synchronous policy. Merge the trained
        # LoRA delta into the base weights once, before loading the auxiliary
        # action/proprio heads or accepting any FastAPI request.
        vla = peft_vla.merge_and_unload(safe_merge=True)
        self._lora_adapter = lora_adapter
        self._lora_merged = True
        self._log_load_event("lora_adapter_merged", safe_merge=True)
        vla.vision_backbone.set_num_images_in_input(self.num_images_in_input)
        with (self.checkpoint / "dataset_statistics.json").open("r", encoding="utf-8") as file:
            vla.norm_stats = json.load(file)
        vla = vla.to(device).eval()

        action_head = L1RegressionActionHead(
            input_dim=vla.llm_dim,
            hidden_dim=vla.llm_dim,
            action_dim=self.action_dim,
        ).to(torch.bfloat16).to(device)
        action_head.load_state_dict(self._load_component(torch, self.checkpoint / "action_head.pt"))
        action_head.eval()

        proprio_projector = ProprioProjector(
            llm_dim=vla.llm_dim,
            proprio_dim=self.action_dim,
        ).to(torch.bfloat16).to(device)
        proprio_projector.load_state_dict(
            self._load_component(torch, self.checkpoint / "proprio_projector.pt")
        )
        proprio_projector.eval()
        self._log_load_event(
            "policy_components_loaded",
            action_head=str(self.checkpoint / "action_head.pt"),
            proprio_projector=str(self.checkpoint / "proprio_projector.pt"),
        )

        self._torch = torch
        self._device = device
        self._vla = vla
        self._processor = processor
        self._action_head = action_head
        self._proprio_projector = proprio_projector
        self._get_vla_action = get_vla_action
        self._cfg = SimpleNamespace(
            use_l1_regression=True,
            use_diffusion=False,
            use_film=self.use_film,
            num_images_in_input=self.num_images_in_input,
            use_proprio=True,
            center_crop=self.image_augmentation,
            unnorm_key=self.unnorm_key,
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    @staticmethod
    def _load_component(torch_module: Any, path: Path) -> dict[str, Any]:
        state = torch_module.load(path, map_location="cpu", weights_only=True)
        return {
            key.removeprefix("module."): value
            for key, value in state.items()
        }

    def predict(self, observation: dict[str, Any], task: str) -> dict[str, Any]:
        state = np.asarray(observation["state"], dtype=np.float32).copy()
        model_observation = self._prepare_observation(observation)
        started = time.perf_counter()
        with self._lock:
            actions = self._get_vla_action(
                self._cfg,
                self._vla,
                self._processor,
                model_observation,
                task,
                self._action_head,
                self._proprio_projector,
            )
            self._torch.cuda.synchronize(self._device)
        guarded, clipped = self.action_safety.apply(actions, state)
        return {
            "actions": guarded.tolist(),
            "action_shape": list(guarded.shape),
            "normalization": self.normalization,
            "clipped_to_training_bounds": clipped,
            # Compatibility for consumers of the first q99-only deployment API.
            "clipped_to_training_q01_q99": clipped if self.normalization == "bounds_q99" else False,
            "inference_ms": (time.perf_counter() - started) * 1000.0,
            "peak_vram_mib": self._torch.cuda.max_memory_allocated(self._device) / 1024**2,
        }

    def health(self) -> dict[str, Any]:
        contract = self._metadata["training_contract"]
        return {
            "ready": True,
            "checkpoint": str(self.checkpoint),
            "checkpoint_source": self.checkpoint_source,
            "checkpoint_source_kind": self.checkpoint_source_kind,
            "checkpoint_step": int(self._metadata["step"]),
            "base_model": str(self.base_model),
            "base_model_source": self.base_model_source,
            "base_model_source_kind": self.base_model_source_kind,
            "lora_adapter": str(self._lora_adapter),
            "lora_merged": self._lora_merged,
            "gpu": self._torch.cuda.get_device_name(self._device),
            "compute_capability": list(self._torch.cuda.get_device_capability(self._device)),
            "action_shape": list(self.action_shape),
            "control_hz": contract.get("control_hz"),
            "normalization": self.normalization,
            "unnorm_key": self.unnorm_key,
            "camera_names": list(self.image_keys),
            "image_preprocessing": self.image_preprocessing,
            "robot_contract": self.robot_contract,
            "resolved_oft_contract": self._resolved_oft_contract,
            "tensorflow_visible_gpus": [],
        }
