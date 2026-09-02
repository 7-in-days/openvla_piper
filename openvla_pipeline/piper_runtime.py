"""Validate the model, robot, topic, shape, and unit runtime contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from openvla_pipeline.model_io import ContractError, validate_action_chunk


ARM_UNIT = "radian"
GRIPPER_UNIT = "meter"


@dataclass(frozen=True)
class PiperModelContract:
    """모델 health와 PiperBridge config에서 배포 주기·shape를 한 번만 계산한다."""

    action_horizon: int
    action_dim: int
    fps: float
    action_keys: tuple[str, ...]
    command_names: tuple[str, ...]
    action_units: tuple[str, ...]
    action_encoding: str
    camera_names: tuple[str, ...]

    @classmethod
    def from_runtime(
        cls,
        model_server_status: dict[str, Any],
        robot_config: Any,
    ) -> "PiperModelContract":
        shape = model_server_status.get("action_shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise ContractError(f"server action_shape must be [horizon, dim], got {shape}")

        horizon, action_dim = (int(shape[0]), int(shape[1]))
        action_keys = tuple(robot_config.joint_keys)
        command_names = tuple(robot_config.command_out_names)
        fps = float(robot_config.fps)

        if horizon <= 0 or action_dim <= 0:
            raise ContractError(f"action_shape values must be positive, got {shape}")
        if action_dim != len(action_keys) or action_dim != len(command_names):
            raise ContractError(
                "model/robot action dimension mismatch: "
                f"model={action_dim}, joint_keys={len(action_keys)}, command_names={len(command_names)}"
            )
        if not np.isfinite(fps) or fps <= 0:
            raise ContractError(f"robot fps must be positive, got {fps}")

        server_hz = model_server_status.get("control_hz")
        if server_hz is not None and not np.isclose(float(server_hz), fps):
            raise ContractError(f"model/robot fps mismatch: model={server_hz}, robot={fps}")

        robot_contract = model_server_status.get("robot_contract")
        action_units = tuple([ARM_UNIT] * (action_dim - 1) + [GRIPPER_UNIT])
        action_encoding = "absolute_joint_position"
        camera_names = tuple(camera[0] for camera in getattr(robot_config, "ros_cameras", ()))
        if robot_contract is not None:
            expected_action_keys = tuple(robot_contract.get("action_names", ()))
            expected_command_names = tuple(robot_contract.get("command_names", ()))
            expected_action_units = tuple(robot_contract.get("action_units", ()))
            expected_state_units = tuple(robot_contract.get("state_units", ()))
            expected_camera_names = tuple(robot_contract.get("camera_names", ()))
            action_encoding = str(robot_contract.get("action_encoding", ""))
            mismatches = {}
            if expected_action_keys != action_keys:
                mismatches["action_names"] = {"model": expected_action_keys, "robot": action_keys}
            if expected_command_names != command_names:
                mismatches["command_names"] = {"model": expected_command_names, "robot": command_names}
            if expected_action_units != action_units or expected_state_units != action_units:
                mismatches["units"] = {
                    "model_action": expected_action_units,
                    "model_state": expected_state_units,
                    "robot": action_units,
                }
            if action_encoding != "absolute_joint_position":
                mismatches["action_encoding"] = {
                    "model": action_encoding,
                    "robot": "absolute_joint_position",
                }
            if set(expected_camera_names) != set(camera_names):
                mismatches["camera_names"] = {"model": expected_camera_names, "robot": camera_names}
            if mismatches:
                raise ContractError(f"model/robot semantic contract mismatch: {mismatches}")
            action_units = expected_action_units
            camera_names = expected_camera_names

        return cls(
            horizon,
            action_dim,
            fps,
            action_keys,
            command_names,
            action_units,
            action_encoding,
            camera_names,
        )
    @property
    def action_shape(self) -> tuple[int, int]:
        return self.action_horizon, self.action_dim

    @property
    def control_interval_s(self) -> float:
        return 1.0 / self.fps

    @property
    def chunk_duration_s(self) -> float:
        return self.action_horizon / self.fps

    def describe(self, input_topic: str, output_topic: str) -> str:
        return json.dumps(
            {
                "input_topic": input_topic,
                "output_topic": output_topic,
                "action_shape": list(self.action_shape),
                "control_hz": self.fps,
                "control_interval_s": self.control_interval_s,
                "chunk_duration_s": self.chunk_duration_s,
                "arm_unit": ARM_UNIT,
                "gripper_unit": GRIPPER_UNIT,
                "action_names": list(self.action_keys),
                "command_names": list(self.command_names),
                "action_units": list(self.action_units),
                "action_encoding": self.action_encoding,
                "camera_names": list(self.camera_names),
            },
            ensure_ascii=False,
        )


# Compatibility alias for the pre-refactor public name.
RuntimeContract = PiperModelContract


def validate_execution_mode(
    motion_enabled: bool,
    async_prefetch_enabled: bool,
    live_confirmation: str | None,
    live_motion_allowed: bool = True,
) -> None:
    """검증되지 않은 async physical 조합을 robot connect 전에 차단한다."""
    if motion_enabled and async_prefetch_enabled:
        raise ContractError(
            "async prefetch is dry-run only; physical motion requires synchronous inference"
        )
    if motion_enabled and not live_motion_allowed:
        raise ContractError(
            "physical motion is disabled by safety.allow_live_motion in the runtime config"
        )
    if motion_enabled and live_confirmation != "YES":
        raise ContractError("physical motion requires PIPER_OPENVLA_LIVE_CONFIRMED=YES")


def validate_robot_units(
    state: Any,
    action_dim: int,
    gripper_index: int,
    gripper_min_m: float = 0.0,
    gripper_max_m: float = 0.085,
) -> np.ndarray:
    """joint 1~6은 radian, gripper는 meter인 feedback 계약을 확인한다."""
    vector = np.asarray(state, dtype=np.float32)
    if vector.shape != (action_dim,):
        raise ContractError(f"Piper state must be {action_dim}D, got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ContractError("Piper state contains NaN or Inf")
    if not 0 <= gripper_index < action_dim:
        raise ContractError(f"invalid gripper index: {gripper_index}")

    # PiPER gripper는 meter 단위이며 데이터 수집 범위가 0~0.085 m이다.
    if not gripper_min_m <= float(vector[gripper_index]) <= gripper_max_m:
        raise ContractError(
            "gripper feedback is outside meter contract: "
            f"{vector[gripper_index]:.6f} m range=[{gripper_min_m}, {gripper_max_m}]"
        )
    return vector


def validate_action_handoff(
    actions: Any,
    previous_action: Any | None,
    expected_shape: tuple[int, int],
    max_arm_delta_rad: float = 1.5,
) -> np.ndarray:
    """이전 chunk의 마지막 absolute target과 새 chunk 첫 target을 대조한다."""
    chunk = validate_action_chunk(actions, expected_shape)
    if previous_action is None:
        return chunk
    previous = np.asarray(previous_action, dtype=np.float32)
    if previous.shape != (expected_shape[1],) or not np.isfinite(previous).all():
        raise ContractError(
            f"previous Piper action must be finite {expected_shape[1]}D, got {previous.shape}"
        )
    arm_delta = float(np.max(np.abs(chunk[0, :-1] - previous[:-1])))
    if arm_delta > max_arm_delta_rad:
        raise ContractError(
            "chunk handoff is discontinuous: "
            f"arm_delta={arm_delta:.6f} rad limit={max_arm_delta_rad:.6f} rad"
        )
    return chunk
