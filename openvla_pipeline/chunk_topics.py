"""Publish action-chunk diagnostic messages on Piper ROS topics."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


CHUNK_TOPIC = "/piper/inference/chunk"
AGGREGATED_CHUNK_TOPIC = "/piper/inference/aggregated_chunk"
MESSAGE_TYPE = "trajectory_msgs/JointTrajectory"


def action_frame_id(chunk_id: int, step_index: int) -> str:
    if chunk_id < 0 or step_index < 0:
        raise ValueError("chunk_id and step_index must be non-negative")
    return f"{chunk_id}:{step_index:02d}"


def trajectory_message(
    actions: Any,
    chunk_id: int,
    command_names: tuple[str, ...],
    fps: float,
    stamp_ns: int,
    start_step: int = 0,
) -> dict[str, Any]:
    matrix = np.asarray(actions, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(command_names):
        raise ValueError(
            f"chunk diagnostic shape mismatch: actions={matrix.shape}, names={len(command_names)}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("chunk diagnostic contains NaN or Inf")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    step_ns = round(1_000_000_000 / fps)
    points = []
    for index, positions in enumerate(matrix.tolist()):
        offset_ns = (start_step + index) * step_ns
        points.append(
            {
                "positions": positions,
                "velocities": [],
                "accelerations": [],
                "effort": [],
                "time_from_start": {
                    "sec": offset_ns // 1_000_000_000,
                    "nanosec": offset_ns % 1_000_000_000,
                },
            }
        )
    return {
        "header": {
            "stamp": {
                "sec": stamp_ns // 1_000_000_000,
                "nanosec": stamp_ns % 1_000_000_000,
            },
            "frame_id": f"chunk_{chunk_id}",
        },
        "joint_names": list(command_names),
        "points": points,
    }


class ActionChunkTopicPublisher:
    """Publish the two existing vla-piper diagnostic topics on Robot's rosbridge."""

    def __init__(self, ros: Any, command_names: tuple[str, ...], fps: float) -> None:
        import roslibpy

        self._roslibpy = roslibpy
        self._command_names = command_names
        self._fps = fps
        self._chunk = roslibpy.Topic(ros, CHUNK_TOPIC, MESSAGE_TYPE, queue_size=10)
        self._aggregated = roslibpy.Topic(
            ros,
            AGGREGATED_CHUNK_TOPIC,
            MESSAGE_TYPE,
            queue_size=10,
        )
        self._closed = False
        chunk_advertised = False
        try:
            self._chunk.advertise()
            chunk_advertised = True
            self._aggregated.advertise()
        except Exception:
            if chunk_advertised:
                self._chunk.unadvertise()
            raise

    def publish(self, actions: Any, chunk_id: int) -> tuple[int, int]:
        original_stamp_ns = time.time_ns()
        original = trajectory_message(
            actions,
            chunk_id,
            self._command_names,
            self._fps,
            original_stamp_ns,
        )
        self._chunk.publish(self._roslibpy.Message(original))

        # Synchronous OpenVLA has no overlapping action queue: effective/aggregated
        # chunk is exactly the original chunk (offset=0, N=H).
        aggregated_stamp_ns = max(time.time_ns(), original_stamp_ns + 1_000)
        aggregated = trajectory_message(
            actions,
            chunk_id,
            self._command_names,
            self._fps,
            aggregated_stamp_ns,
        )
        self._aggregated.publish(self._roslibpy.Message(aggregated))
        return original_stamp_ns, aggregated_stamp_ns

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._chunk.unadvertise()
        self._aggregated.unadvertise()


# Compatibility alias for pre-refactor imports.
ChunkDiagnosticsPublisher = ActionChunkTopicPublisher
