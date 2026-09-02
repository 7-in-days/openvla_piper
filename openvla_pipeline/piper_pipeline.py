"""Run the synchronous OpenVLA observation-to-Piper-action pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from openvla_pipeline import user_settings
from openvla_pipeline.model_io import (
    LIVE_CONFIRMATION,
    observation_to_request,
    validate_action_chunk,
)
from openvla_pipeline.piper_dry_run import request_json
from openvla_async_pipeline.async_chunk_prefetcher import AsyncChunkPrefetcher, PreparedChunk
from openvla_pipeline.config import load_runtime_config
from openvla_pipeline.chunk_topics import ActionChunkTopicPublisher, action_frame_id
from openvla_pipeline.inference_logger import InferenceSessionLogger
from openvla_pipeline.piper_runtime import (
    PiperModelContract,
    validate_action_handoff,
    validate_execution_mode,
    validate_robot_units,
)


DEFAULT_TASK = user_settings.TASK


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, on/off; got {raw!r}")


@dataclass(frozen=True)
class PiperPipelineConfig:
    task: str
    motion_enabled: bool
    max_actions: int
    async_prefetch_enabled: bool
    chunk_wait_timeout_s: float
    max_chunk_age_s: float
    model_server: str
    rosbridge_url: str
    piper_repo: Path
    inference_log_root: Path
    session_id: str | None
    chunk_diagnostics_enabled: bool
    health_timeout_s: float
    request_timeout_s: float
    auth_token_env: str
    allow_live_motion: bool
    gripper_min_m: float
    gripper_max_m: float
    max_arm_step_delta_rad: float


def parse_settings(argv: list[str] | None = None) -> PiperPipelineConfig:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args(argv)
    config = load_runtime_config(config_args.config)

    default_mode = os.environ.get("PIPER_OPENVLA_MODE")
    if default_mode is None:
        default_mode = "dry-run" if _env_bool("PIPER_OPENVLA_DRY_RUN", True) else "live"
    parser = argparse.ArgumentParser(description="OpenVLA server → existing Piper ROS topics")
    parser.add_argument("--config", type=Path, default=config.source_path)
    parser.add_argument("--mode", choices=("dry-run", "live"), default=default_mode)
    parser.add_argument(
        "--task",
        default=os.environ.get("PIPER_OPENVLA_TASK", config.client.task),
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=int(os.environ.get("PIPER_OPENVLA_MAX_ACTIONS", config.client.max_actions)),
    )
    parser.add_argument(
        "--async-prefetch",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("PIPER_OPENVLA_ASYNC_PREFETCH", False),
    )
    parser.add_argument("--chunk-wait-timeout-s", type=float, default=3.0)
    parser.add_argument("--max-chunk-age-s", type=float, default=2.0)
    parser.add_argument(
        "--server",
        default=os.environ.get("PIPER_OPENVLA_SERVER", config.client.model_server),
    )
    parser.add_argument(
        "--rosbridge-url",
        default=os.environ.get("PIPER_ROSBRIDGE_URL", config.client.rosbridge_url),
    )
    parser.add_argument(
        "--piper-repo",
        type=Path,
        default=Path(os.environ.get("PIPER_REPO", config.client.piper_repo)),
    )
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--session-id", default=os.environ.get("PIPER_INFERENCE_SESSION_ID"))
    parser.add_argument(
        "--chunk-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=_env_bool(
            "PIPER_OPENVLA_CHUNK_DIAGNOSTICS",
            config.client.chunk_diagnostics,
        ),
    )
    parser.add_argument(
        "--health-timeout-s",
        type=float,
        default=float(
            os.environ.get(
                "PIPER_OPENVLA_HEALTH_TIMEOUT_S", config.client.health_timeout_s
            )
        ),
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=float(
            os.environ.get(
                "PIPER_OPENVLA_REQUEST_TIMEOUT_S", config.client.request_timeout_s
            )
        ),
    )
    args = parser.parse_args(argv)
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")
    if (
        args.chunk_wait_timeout_s <= 0
        or args.max_chunk_age_s <= 0
        or args.health_timeout_s <= 0
        or args.request_timeout_s <= 0
    ):
        parser.error("chunk timeouts must be positive")
    log_root = args.log_root or Path(
        os.environ.get("PIPER_INFERENCE_LOG_ROOT", config.client.inference_log_root)
    )
    return PiperPipelineConfig(
        task=args.task,
        motion_enabled=args.mode == "live",
        max_actions=args.max_actions,
        async_prefetch_enabled=args.async_prefetch,
        chunk_wait_timeout_s=args.chunk_wait_timeout_s,
        max_chunk_age_s=args.max_chunk_age_s,
        model_server=args.server.rstrip("/"),
        rosbridge_url=args.rosbridge_url,
        piper_repo=args.piper_repo,
        inference_log_root=log_root,
        session_id=args.session_id,
        chunk_diagnostics_enabled=args.chunk_diagnostics,
        health_timeout_s=args.health_timeout_s,
        request_timeout_s=args.request_timeout_s,
        auth_token_env=config.server.auth_token_env,
        allow_live_motion=config.safety.allow_live_motion,
        gripper_min_m=config.safety.gripper_min_m,
        gripper_max_m=config.safety.gripper_max_m,
        max_arm_step_delta_rad=config.safety.max_arm_step_delta_rad,
    )


class PiperOpenVLAPipeline:
    """vla-piper와 같은 SyncedFrame 입력 → JointState 출력 파이프라인이다."""

    def __init__(self, config: PiperPipelineConfig):
        self.config = config
        self.task = config.task
        self.motion_enabled = config.motion_enabled
        self.async_prefetch_enabled = config.async_prefetch_enabled
        self._check_live_gate()

        piper_repo = config.piper_repo.resolve()
        sys.path.insert(0, str(piper_repo))

        from piper_bridge.config_piper_bridge import PiperBridgeRobotConfig
        from piper_bridge.piper_bridge_robot import PiperBridgeRobot

        self.action_count = 0
        self.chunk_id = 0
        self.previous_action = None
        self.pending_request_logs: dict[str, dict[str, object]] = {}
        self.termination_reason = "normal"

        self.token = os.environ.get(config.auth_token_env) or None
        self.server = config.model_server

        # 토픽명·fps·관절 키는 기존 vla-piper config를 단일 기준으로 사용한다.
        self.piper_bridge_config = PiperBridgeRobotConfig(
            rosbridge_url=config.rosbridge_url
        )
        self.model_server_status = self._read_model_health()
        self.piper_model_contract = PiperModelContract.from_runtime(
            self.model_server_status,
            self.piper_bridge_config,
        )
        self.gripper_index = self.piper_model_contract.action_keys.index("gripper.pos")

        self.runtime_description = json.loads(
            self.piper_model_contract.describe(
                self.piper_bridge_config.frame_topic,
                self.piper_bridge_config.output_topic,
            )
        )
        print("[OpenVLA] Piper ROS bridge 연결 중")
        self.piper_robot = PiperBridgeRobot(self.piper_bridge_config)
        self.action_chunk_topics = None
        try:
            self.piper_robot.connect()
            print("[OpenVLA] Piper ROS bridge 연결 완료")
            self.action_chunk_topics = (
                ActionChunkTopicPublisher(
                    self.piper_robot._ros,
                    self.piper_model_contract.command_names,
                    self.piper_model_contract.fps,
                )
                if config.chunk_diagnostics_enabled
                else None
            )
            self.inference_logger = InferenceSessionLogger(
                config.inference_log_root,
                config.session_id,
                {
                    "client": {
                        "mode": "live" if self.motion_enabled else "dry-run",
                        "task": self.task,
                        "max_actions": config.max_actions,
                        "async_prefetch": self.async_prefetch_enabled,
                        "server": self.server,
                        "rosbridge_url": config.rosbridge_url,
                    },
                    "topic_contract": self.runtime_description,
                    "server_health": self.model_server_status,
                },
                core_enabled=config.chunk_diagnostics_enabled,
            )
        except Exception:
            if self.action_chunk_topics is not None:
                self.action_chunk_topics.close()
            if self.piper_robot.is_connected:
                self.piper_robot.disconnect()
            raise
        print(
            "[OpenVLA] runtime contract | "
            + self.piper_model_contract.describe(
                self.piper_bridge_config.frame_topic,
                self.piper_bridge_config.output_topic,
            )
        )
        print(f"[OpenVLA] mode={'live' if self.motion_enabled else 'dry-run'} | task={self.task}")
        print(f"[OpenVLA] inference log | {self.inference_logger.session_dir}")
        self.prefetcher = (
            AsyncChunkPrefetcher(
                self._infer_one_chunk,
                self.piper_model_contract.action_shape,
            )
            if self.async_prefetch_enabled
            else None
        )

    # ========== 시작 전 검사 ==========

    def _check_live_gate(self):
        validate_execution_mode(
            self.motion_enabled,
            self.async_prefetch_enabled,
            os.environ.get(LIVE_CONFIRMATION),
            self.config.allow_live_motion,
        )

    def _read_model_health(self) -> dict:
        health = request_json(
            f"{self.server}/health",
            None,
            self.token,
            timeout=self.config.health_timeout_s,
        )
        if health.get("ready") is not True:
            raise RuntimeError(f"OpenVLA server가 준비되지 않음: {health}")
        return health

    # ========== OpenVLA 연속 추론 ==========

    def run(self):
        try:
            if self.prefetcher is None:
                self._run_synchronous()
            else:
                self._run_async_dry_run()
        except KeyboardInterrupt:
            self.termination_reason = "keyboard_interrupt"
            print("[OpenVLA] 사용자 정지")
        except Exception:
            self.termination_reason = "error"
            raise
        finally:
            self.close()

    def _run_synchronous(self):
        while self.action_count < self.config.max_actions:
            requested_at = time.monotonic()
            response = self._infer_one_chunk(self.chunk_id)
            try:
                chunk = PreparedChunk(
                    sequence=self.chunk_id,
                    request_id=str(response["request_id"]),
                    actions=validate_action_chunk(
                        response.get("actions"),
                        self.piper_model_contract.action_shape,
                    ).copy(),
                    server_inference_ms=float(response["inference_ms"]),
                    requested_at=requested_at,
                    ready_at=time.monotonic(),
                )
            except Exception as exc:
                self._record_failure(
                    self.chunk_id,
                    str(response.get("request_id", "")),
                    "response_contract_rejected",
                    exc,
                )
                raise
            self._run_one_chunk(chunk)

    def _run_async_dry_run(self):
        assert self.prefetcher is not None
        self.prefetcher.request()
        while self.action_count < self.config.max_actions:
            chunk = self.prefetcher.get(
                self.config.chunk_wait_timeout_s,
                self.config.max_chunk_age_s,
            )
            if self.action_count + len(chunk.actions) < self.config.max_actions:
                self.prefetcher.request()
            self._run_one_chunk(chunk)

    def _infer_one_chunk(self, sequence: int) -> dict:
        observation_stream_status = self.piper_robot.get_stream_status()
        if not observation_stream_status.get("ready"):
            raise RuntimeError(
                "Piper 관측 stream이 준비되지 않음: "
                f"{observation_stream_status}"
            )

        observation_packet = {}
        packet_getter = getattr(self.piper_robot, "get_observation_packet", None)
        if packet_getter is None:
            observation = self.piper_robot.get_observation()
        else:
            observation, observation_packet = packet_getter()
        state = np.asarray(
            [observation[key] for key in self.piper_model_contract.action_keys],
            dtype=np.float32,
        )
        validate_robot_units(
            state,
            self.piper_model_contract.action_dim,
            self.gripper_index,
            self.config.gripper_min_m,
            self.config.gripper_max_m,
        )

        request_id = f"{sequence}-{uuid.uuid4().hex}"
        request = observation_to_request(
            observation,
            self.task,
            request_id,
            action_keys=self.piper_model_contract.action_keys,
            image_keys=self.piper_model_contract.camera_names,
        )
        request_stamp_ns = time.time_ns()
        try:
            response = request_json(
                f"{self.server}/act",
                request,
                self.token,
                timeout=self.config.request_timeout_s,
            )
        except Exception as exc:
            self._record_failure(sequence, request_id, "send_failed", exc, request_stamp_ns)
            raise
        response_received_stamp_ns = time.time_ns()
        if response.get("request_id") != request_id:
            error = RuntimeError("OpenVLA response request_id가 요청과 다름")
            self._record_failure(
                sequence,
                request_id,
                "response_id_mismatch",
                error,
                request_stamp_ns,
            )
            raise error
        self.pending_request_logs[request_id] = {
            "request_stamp_ns": request_stamp_ns,
            "response_received_stamp_ns": response_received_stamp_ns,
            "observation_packet": observation_packet,
            "response": {
                "action_shape": response.get("action_shape"),
                "normalization": response.get("normalization"),
                "clipped_to_training_bounds": response.get(
                    "clipped_to_training_bounds"
                ),
                "peak_vram_mib": response.get("peak_vram_mib"),
            },
        }
        return response

    def _record_failure(
        self,
        sequence: int,
        request_id: str,
        phase: str,
        error: BaseException,
        request_stamp_ns: int | None = None,
    ) -> None:
        if request_stamp_ns is None:
            pending = self.pending_request_logs.get(request_id, {})
            pending_stamp = pending.get("request_stamp_ns")
            request_stamp_ns = None if pending_stamp is None else int(pending_stamp)
        self.inference_logger.append_observability(
            {
                "event": "request_failure",
                "schema_version": 1,
                "request_sequence": int(sequence),
                "request_id": request_id or None,
                "request_stamp_ns": request_stamp_ns,
                "failure_phase": phase,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )

    def _run_one_chunk(self, chunk: PreparedChunk):
        try:
            actions = validate_action_handoff(
                chunk.actions,
                self.previous_action,
                self.piper_model_contract.action_shape,
                self.config.max_arm_step_delta_rad,
            )
        except Exception as exc:
            self._record_failure(
                chunk.sequence, chunk.request_id, "chunk_handoff_rejected", exc
            )
            self.pending_request_logs.pop(chunk.request_id, None)
            raise
        request_log = self.pending_request_logs.pop(chunk.request_id)
        original_stamp_ns = None
        aggregated_stamp_ns = None
        if self.action_chunk_topics is not None:
            try:
                original_stamp_ns, aggregated_stamp_ns = self.action_chunk_topics.publish(
                    actions,
                    self.chunk_id,
                )
            except Exception as exc:
                self._record_failure(
                    chunk.sequence, chunk.request_id, "chunk_topic_publish", exc
                )
                raise
            self.inference_logger.append_chunk(
                chunk_id=self.chunk_id,
                request_stamp_ns=int(request_log["request_stamp_ns"]),
                observation_packet=dict(request_log["observation_packet"]),
                instruction=self.task,
                publish_stamp_ns=original_stamp_ns,
            )
        response_metadata = dict(request_log.get("response", {}))
        observation_packet = dict(request_log["observation_packet"])
        self.inference_logger.append_observability(
            {
                "event": "chunk",
                "schema_version": 1,
                "chunk_id": int(self.chunk_id),
                "request_id": chunk.request_id,
                "request_stamp_ns": int(request_log["request_stamp_ns"]),
                "response_received_stamp_ns": int(
                    request_log["response_received_stamp_ns"]
                ),
                "obs_batch_stamp_ns": observation_packet.get("obs_batch_stamp_ns"),
                "obs_received_stamp_ns": observation_packet.get(
                    "obs_received_stamp_ns"
                ),
                "publish_stamp_ns": original_stamp_ns,
                "aggregated_publish_stamp_ns": aggregated_stamp_ns,
                "server_model_inference_ms": float(chunk.server_inference_ms),
                "client_inference_wall_ms": float(chunk.inference_wall_ms),
                "action_shape": list(actions.shape),
                "mode": "live" if self.motion_enabled else "dry-run",
                **response_metadata,
            }
        )
        print(
            f"[OpenVLA] chunk={self.chunk_id} | shape={actions.shape} "
            f"| horizon={self.piper_model_contract.chunk_duration_s:.3f}s "
            f"| inference={chunk.server_inference_ms:.1f}ms "
            f"| wall={chunk.inference_wall_ms:.1f}ms "
            f"| units=rad×{self.piper_model_contract.action_dim - 1},m×1"
        )

        for step_index, action in enumerate(actions):
            if self.action_count >= self.config.max_actions:
                break

            tick_started = time.perf_counter()
            if self.motion_enabled:
                self.piper_robot._piper_frame_id_provider = (
                    lambda chunk_id=self.chunk_id, step=step_index: action_frame_id(chunk_id, step)
                )
                robot_action = {
                    key: float(value)
                    for key, value in zip(
                        self.piper_model_contract.action_keys,
                        action,
                        strict=True,
                    )
                }
                try:
                    self.piper_robot.send_action(robot_action)
                except Exception as exc:
                    self._record_failure(
                        chunk.sequence,
                        chunk.request_id,
                        "robot_output_publish",
                        exc,
                    )
                    raise

            self.previous_action = action.copy()
            self.action_count += 1
            elapsed = time.perf_counter() - tick_started
            time.sleep(
                max(0.0, self.piper_model_contract.control_interval_s - elapsed)
            )

        self.chunk_id += 1

    # ========== 종료 ==========

    def close(self):
        if self.prefetcher is not None:
            self.prefetcher.close()
        if self.action_chunk_topics is not None:
            self.action_chunk_topics.close()
        if hasattr(self.piper_robot, "_piper_frame_id_provider"):
            delattr(self.piper_robot, "_piper_frame_id_provider")
        if self.piper_robot.is_connected:
            self.piper_robot.disconnect()
        self.inference_logger.close(
            self.termination_reason,
            self.action_count,
            self.chunk_id,
        )
        print(f"[OpenVLA] 종료 | actions={self.action_count} | chunks={self.chunk_id}")


def main(argv: list[str] | None = None):
    pipeline = PiperOpenVLAPipeline(parse_settings(argv))
    pipeline.run()


# Compatibility aliases for pre-refactor imports.
TopicNodeSettings = PiperPipelineConfig
PiperOpenVLATopicNode = PiperOpenVLAPipeline


if __name__ == "__main__":
    main()
