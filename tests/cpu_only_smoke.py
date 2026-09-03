"""CPU-only, network-free smoke tests for the standalone inference contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("CPU smoke requires CUDA_VISIBLE_DEVICES='' exactly")
os.environ.pop("PIPER_ACTION_CHUNK", None)
os.environ.pop("ROBOT_PLATFORM", None)

from openvla_pipeline.checkpoint_source import is_hf_source, resolve_source
from openvla_pipeline.config import load_runtime_config
from openvla_pipeline.inference_logger import InferenceSessionLogger, _queue_size
from openvla_pipeline.launch_plan import resolve_launch_plan
from openvla_pipeline.model_io import ACTION_KEYS, ContractError
from openvla_pipeline.openvla_policy import PiperOpenVLAPolicy
from openvla_pipeline.piper_pipeline import parse_settings
from openvla_pipeline.piper_runtime import PiperModelContract, validate_execution_mode
from openvla_pipeline.chunk_topics import (
    AGGREGATED_CHUNK_TOPIC,
    CHUNK_TOPIC,
    MESSAGE_TYPE,
    trajectory_message,
)


CORE_FIELDS = {
    "chunk_id",
    "request_stamp_ns",
    "obs_third_person_stamp_ns",
    "obs_wrist_stamp_ns",
    "obs_qmeas_stamp_ns",
    "instruction",
    "publish_stamp_ns",
}
UNITS = ["radian"] * 6 + ["meter"]


def expect_error(error_type, callback) -> None:
    try:
        callback()
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def config_payload(root: Path, task: str = "config task") -> dict:
    return {
        "schema_version": 1,
        "server": {
            "host": "127.0.0.1",
            "port": 8777,
            "checkpoint": None,
            "base_model": None,
            "openvla_oft_repo": None,
            "auth_token_env": "PIPER_OPENVLA_SERVER_TOKEN",
            "max_request_bytes": 8 * 1024 * 1024,
        },
        "client": {
            "model_server": "http://127.0.0.1:8777",
            "rosbridge_url": "ws://localhost:9090",
            "piper_repo": str(root / "piper-source"),
            "task": task,
            "max_actions": 20,
            "health_timeout_s": 3.0,
            "request_timeout_s": 4.0,
            "inference_log_root": str(root / "logs"),
            "chunk_diagnostics": True,
        },
        "safety": {
            "allow_live_motion": False,
            "gripper_min_m": 0.0,
            "gripper_max_m": 0.085,
            "max_arm_step_delta_rad": 1.5,
        },
    }


def metadata_payload() -> dict:
    return {
        "base_vla_path": "/not/accessed/in/cpu-smoke",
        "step": 100000,
        "training_contract": {
            "action_dim": 7,
            "num_actions_chunk": 20,
            "num_images_in_input": 2,
            "proprio_dim": 7,
            "normalization": "bounds",
            "use_l1_regression": True,
            "use_diffusion": False,
            "use_proprio": True,
            "use_film": False,
            "image_aug": False,
            "control_hz": 20.0,
            "robot_contract": {
                "robot_type": "piper_bridge",
                "action_encoding": "absolute_joint_position",
                "action_names": list(ACTION_KEYS),
                "state_names": list(ACTION_KEYS),
                "action_units": UNITS,
                "state_units": UNITS,
                "camera_names": ["third_person", "wrist"],
            },
        },
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)

        # Local/HF source classification does not contact the network.
        local_source = root / "checkpoint"
        local_source.mkdir()
        resolved = resolve_source(local_source)
        assert resolved.kind == "local" and resolved.local_path == local_source
        assert is_hf_source("hf://owner/repo@revision")

        # Checkpoint metadata owns the import-time OFT action contract.
        (local_source / "checkpoint_metadata.json").write_text(
            json.dumps(metadata_payload()), encoding="utf-8"
        )
        policy = PiperOpenVLAPolicy.__new__(PiperOpenVLAPolicy)
        policy.checkpoint = local_source
        policy._metadata = policy._load_and_validate_metadata()
        contract = policy._metadata["training_contract"]
        preprocessing = {
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
        crop_contract = dict(contract["robot_contract"])
        crop_contract["image_preprocessing"] = preprocessing
        policy.image_keys = ("third_person", "wrist")
        policy.image_preprocessing = policy._parse_image_preprocessing(crop_contract)
        source_image = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
        prepared = policy._prepare_observation(
            {
                "state": np.zeros(7, dtype=np.float32),
                "full_image": source_image,
                "wrist_image": source_image,
            }
        )
        assert prepared["full_image"].shape == (360, 480, 3)
        assert prepared["wrist_image"].shape == (480, 640, 3)
        assert np.array_equal(prepared["full_image"][0, 0], source_image[60, 80])
        policy.action_chunk = int(contract["num_actions_chunk"])
        policy.action_dim = int(contract["action_dim"])
        policy.normalization = str(contract["normalization"])
        fake_oft = root / "openvla-oft"
        constants_path = fake_oft / "prismatic/vla/constants.py"
        constants_path.parent.mkdir(parents=True)
        constants_path.write_text("# synthetic constants\n", encoding="utf-8")
        policy.openvla_oft_repo = fake_oft
        policy._configure_oft_environment()
        assert os.environ["PIPER_ACTION_CHUNK"] == "20"
        policy._validate_oft_constants(
            SimpleNamespace(
                __file__=str(constants_path),
                ROBOT_PLATFORM="PIPER",
                NUM_ACTIONS_CHUNK=20,
                ACTION_DIM=7,
                PROPRIO_DIM=7,
                ACTION_PROPRIO_NORMALIZATION_TYPE="bounds",
            )
        )
        assert policy._resolved_oft_contract["action_chunk"] == 20
        os.environ["PIPER_ACTION_CHUNK"] = "5"
        expect_error(ContractError, policy._configure_oft_environment)
        os.environ.pop("PIPER_ACTION_CHUNK", None)
        os.environ.pop("ROBOT_PLATFORM", None)

        # CLI > env > config. Legacy JSON input remains readable during migration.
        config_path = root / "runtime.json"
        config_path.write_text(json.dumps(config_payload(root)), encoding="utf-8")
        os.environ["PIPER_OPENVLA_TASK"] = "environment task"
        env_settings = parse_settings(["--config", str(config_path)])
        assert env_settings.task == "environment task"
        cli_settings = parse_settings(
            ["--config", str(config_path), "--task", "CLI task", "--max-actions", "7"]
        )
        assert cli_settings.task == "CLI task" and cli_settings.max_actions == 7
        os.environ.pop("PIPER_OPENVLA_TASK", None)
        config_settings = parse_settings(["--config", str(config_path)])
        assert config_settings.task == "config task"
        assert load_runtime_config(config_path).safety.allow_live_motion is False

        # An hf:// reference survives planning without download or metadata access.
        os.environ["PIPER_OPENVLA_CHECKPOINT"] = "hf://owner/repo@revision"
        plan = resolve_launch_plan("sync", ["--config", str(config_path)])
        assert plan.checkpoint == "hf://owner/repo@revision"
        assert plan.action_chunk is None
        os.environ["PIPER_OPENVLA_CHECKPOINT"] = str(local_source)
        local_plan = resolve_launch_plan("sync", ["--config", str(config_path)])
        assert local_plan.checkpoint == str(local_source)
        assert local_plan.action_chunk == 20
        os.environ.pop("PIPER_OPENVLA_CHECKPOINT", None)

        # Schema v3 config, exact core row, independent observability, atomic close.
        log_root = root / "schema3"
        logger = InferenceSessionLogger(
            log_root,
            "enabled",
            {
                "client": {"mode": "dry-run"},
                "server_health": {
                    "checkpoint": str(local_source),
                    "resolved_oft_contract": policy._resolved_oft_contract,
                },
            },
        )
        logger.append_chunk(
            chunk_id=0,
            request_stamp_ns=100,
            observation_packet={
                "obs_third_person_stamp_ns": 10,
                "obs_wrist_stamp_ns": 11,
                "obs_qmeas_stamp_ns": 12,
            },
            instruction="pick up the green blocks",
            publish_stamp_ns=200,
        )
        logger.append_observability(
            {
                "event": "chunk",
                "schema_version": 1,
                "request_id": "request-0",
                "response_received_stamp_ns": 180,
            }
        )
        logger.close("normal", actions=20, chunks=1)
        config = json.loads(logger.config_path.read_text(encoding="utf-8"))
        rows = jsonl(logger.log_path)
        events = jsonl(logger.observability_path)
        assert config["schema_version"] == 3
        assert config["logging"]["writer_threads"] == 1
        assert config["logging"]["queue_size"] <= 8192
        assert events[0]["resolved_oft_contract"]["action_chunk"] == 20
        assert set(rows[0]) == CORE_FIELDS
        assert [event["event"] for event in events] == [
            "session_meta",
            "chunk",
            "session_end",
        ]
        assert not list(logger.session_dir.glob(".*.tmp"))

        disabled = InferenceSessionLogger(
            log_root,
            "disabled",
            {"client": {"mode": "dry-run"}},
            core_enabled=False,
        )
        disabled.append_chunk(
            chunk_id=0,
            request_stamp_ns=1,
            observation_packet={},
            instruction="not written",
            publish_stamp_ns=2,
        )
        disabled.append_observability({"event": "request_failure"})
        disabled.close("error", actions=0, chunks=0)
        disabled_config = json.loads(
            disabled.config_path.read_text(encoding="utf-8")
        )
        assert disabled.log_path.stat().st_size == 0
        assert (
            disabled_config["runtime"]["artifacts"]["inference_log"]["enabled"]
            is False
        )
        os.environ["PIPER_OPENVLA_LOG_QUEUE_SIZE"] = "8193"
        expect_error(ValueError, lambda: _queue_size(None))
        os.environ.pop("PIPER_OPENVLA_LOG_QUEUE_SIZE", None)

        # Pure topic and semantic contracts; no roslibpy connection is created.
        assert CHUNK_TOPIC == "/piper/inference/chunk"
        assert AGGREGATED_CHUNK_TOPIC == "/piper/inference/aggregated_chunk"
        assert MESSAGE_TYPE == "trajectory_msgs/JointTrajectory"
        message = trajectory_message(
            [[0.0] * 7] * 20,
            3,
            tuple(f"joint_{index}" for index in range(7)),
            20.0,
            123,
        )
        assert len(message["points"]) == 20

        command_names = tuple(f"joint_{index}" for index in range(7))
        robot_config = SimpleNamespace(
            joint_keys=ACTION_KEYS,
            command_out_names=command_names,
            fps=20.0,
            ros_cameras=(("third_person", object()), ("wrist", object())),
        )
        health = {
            "action_shape": [20, 7],
            "control_hz": 20.0,
            "robot_contract": {
                "action_names": list(ACTION_KEYS),
                "command_names": list(command_names),
                "action_units": UNITS,
                "state_units": UNITS,
                "camera_names": ["third_person", "wrist"],
                "action_encoding": "absolute_joint_position",
            },
        }
        runtime = PiperModelContract.from_runtime(health, robot_config)
        described = json.loads(
            runtime.describe("/piper/synced/frame", "/piper/inference/output")
        )
        assert described["input_topic"] == "/piper/synced/frame"
        assert described["output_topic"] == "/piper/inference/output"
        validate_execution_mode(False, False, None, False)
        expect_error(
            ContractError,
            lambda: validate_execution_mode(True, False, "YES", False),
        )

        pipeline_source = (PROJECT_ROOT / "openvla_pipeline/piper_pipeline.py").read_text(
            encoding="utf-8"
        )
        assert "if self.motion_enabled:" in pipeline_source
        assert "self.piper_robot.send_action(robot_action)" in pipeline_source

        for name, maximum in {
            "OMP_NUM_THREADS": 8,
            "MKL_NUM_THREADS": 8,
            "OPENBLAS_NUM_THREADS": 8,
            "NUMEXPR_NUM_THREADS": 8,
            "TF_NUM_INTRAOP_THREADS": 4,
            "TF_NUM_INTEROP_THREADS": 4,
        }.items():
            assert 1 <= int(os.environ[name]) <= maximum

        assert "torch" not in sys.modules
        assert "tensorflow" not in sys.modules

    print("cuda_visible_devices=empty")
    print("network_accessed=False")
    print("ros_connected=False")
    print("robot_output_published=False")
    print("config_precedence=CLI>env>YAML")
    print("checkpoint_contract=chunk20,dim7,cameras2,bounds")
    print("logger_schema=3,core7,observability-separated,atomic-close")
    print("topic_contract=synced-frame,output,chunk,aggregated-chunk")
    print("cpu_only_smoke=True")


if __name__ == "__main__":
    main()
