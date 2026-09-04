"""OpenVLA model request, observation, and action data contract for Piper."""

from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ACTION_KEYS = (
    "joint_1.pos",
    "joint_2.pos",
    "joint_3.pos",
    "joint_4.pos",
    "joint_5.pos",
    "joint_6.pos",
    "gripper.pos",
)
ACTION_DIM = len(ACTION_KEYS)


def _action_chunk_from_environment() -> int | None:
    raw_value = os.environ.get("PIPER_ACTION_CHUNK")
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"PIPER_ACTION_CHUNK must be a positive integer, got {raw_value!r}"
        ) from exc
    if value <= 0:
        raise RuntimeError(
            f"PIPER_ACTION_CHUNK must be a positive integer, got {raw_value!r}"
        )
    return value


ACTION_CHUNK = _action_chunk_from_environment()
IMAGE_KEYS = ("third_person", "wrist")
MODEL_IMAGE_SIZE = 224
CLIENT_RESIZE_ALGORITHM = "openvla_oft_rlds_lanczos3_224"
UNNORM_KEY = "piper_bridge"
LIVE_CONFIRMATION = "PIPER_OPENVLA_LIVE_CONFIRMED"


class ContractError(ValueError):
    """Raised when a deployment request violates the trained Piper contract."""


def _validate_rgb_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ContractError(f"image must be uint8 HxWx3 RGB, got shape={array.shape} dtype={array.dtype}")
    return array


@lru_cache(maxsize=1)
def _client_tensorflow() -> Any:
    """Initialize TensorFlow once without exposing the robot process to CUDA."""

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        if tf.config.get_visible_devices("GPU"):
            raise RuntimeError(
                "TensorFlow initialized CUDA before client image resize setup"
            ) from exc
    return tf


def resize_image_for_policy(image: np.ndarray, resize_size: int = MODEL_IMAGE_SIZE) -> np.ndarray:
    """Apply the exact OpenVLA-OFT/RLDS inference resize on the client CPU."""

    array = _validate_rgb_image(image)
    if resize_size <= 0:
        raise ContractError(f"resize_size must be positive, got {resize_size}")

    # Kept byte-for-byte equivalent to the pinned OpenVLA-OFT helper at
    # experiments/robot/openvla_utils.py::resize_image_for_policy.
    tf = _client_tensorflow()

    resized = tf.image.encode_jpeg(array)
    resized = tf.io.decode_image(resized, expand_animations=False, dtype=tf.uint8)
    resized = tf.image.resize(
        resized,
        (resize_size, resize_size),
        method="lanczos3",
        antialias=True,
    )
    resized = tf.cast(tf.clip_by_value(tf.round(resized), 0, 255), tf.uint8)
    return resized.numpy()


def warmup_client_image_resize() -> None:
    """Pay TensorFlow import/kernel setup before the first synchronous request."""

    resize_image_for_policy(np.zeros((480, 640, 3), dtype=np.uint8))


def encode_png_bytes(image: np.ndarray) -> bytes:
    array = _validate_rgb_image(image)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def encode_png(image: np.ndarray) -> str:
    return base64.b64encode(encode_png_bytes(image)).decode("ascii")


def decode_png(encoded: str) -> np.ndarray:
    try:
        payload = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "PNG":
                raise ContractError(f"transport image must be PNG, got {image.format}")
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"invalid PNG payload: {exc}") from exc
    return array


def validate_state(state: Any, expected_dim: int = ACTION_DIM) -> np.ndarray:
    array = np.asarray(state, dtype=np.float32)
    if array.shape != (expected_dim,):
        raise ContractError(f"state must have shape ({expected_dim},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ContractError("state contains NaN or Inf")
    return array


def validate_action_chunk(
    actions: Any,
    expected_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float32)
    if expected_shape is None:
        if ACTION_CHUNK is None:
            raise ContractError(
                "expected_shape is required when PIPER_ACTION_CHUNK is not set; "
                "deployment must use the checkpoint action_shape"
            )
        shape = (ACTION_CHUNK, ACTION_DIM)
    else:
        shape = expected_shape
    if array.shape != shape:
        raise ContractError(
            f"actions must have shape {shape}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ContractError("actions contain NaN or Inf")
    return array


@dataclass(frozen=True)
class PiperActionSafetyGuard:
    low: np.ndarray
    high: np.ndarray
    expected_shape: tuple[int, int]
    max_initial_arm_delta_rad: float = 1.5

    @classmethod
    def from_statistics(
        cls,
        statistics_path: Path,
        normalization: str,
        expected_shape: tuple[int, int],
        unnorm_key: str = UNNORM_KEY,
        max_initial_arm_delta_rad: float = 1.5,
    ) -> "PiperActionSafetyGuard":
        with statistics_path.open("r", encoding="utf-8") as file:
            statistics = json.load(file)
        try:
            action_stats = statistics[unnorm_key]["action"]
            if normalization == "bounds":
                low_key, high_key = "min", "max"
            elif normalization == "bounds_q99":
                low_key, high_key = "q01", "q99"
            else:
                raise ContractError(f"unsupported Piper normalization: {normalization}")
            low = np.asarray(action_stats[low_key], dtype=np.float32)
            high = np.asarray(action_stats[high_key], dtype=np.float32)
        except (KeyError, TypeError) as exc:
            raise ContractError(f"invalid dataset statistics at {statistics_path}: {exc}") from exc
        action_dim = expected_shape[1]
        if low.shape != (action_dim,) or high.shape != (action_dim,):
            raise ContractError(
                f"dataset action bounds must both be {action_dim}D, "
                f"got low={low.shape} high={high.shape}"
            )
        return cls(
            low=low,
            high=high,
            expected_shape=expected_shape,
            max_initial_arm_delta_rad=max_initial_arm_delta_rad,
        )

    def apply(self, actions: Any, state: Any) -> tuple[np.ndarray, bool]:
        chunk = validate_action_chunk(actions, self.expected_shape)
        current = validate_state(state, self.expected_shape[1])
        arm_targets = np.concatenate((current[None, :6], chunk[:, :6]), axis=0)
        max_delta = float(np.max(np.abs(np.diff(arm_targets, axis=0))))
        if max_delta > self.max_initial_arm_delta_rad:
            raise ContractError(
                "predicted arm target is too far from the previous target: "
                f"max_delta={max_delta:.6f} rad limit={self.max_initial_arm_delta_rad:.6f} rad"
            )
        clipped = np.clip(chunk, self.low, self.high)
        return clipped, not np.array_equal(clipped, chunk)


# Compatibility alias for the pre-refactor public name.
ActionGuard = PiperActionSafetyGuard


def observation_to_request(
    observation: dict[str, Any],
    task: str,
    request_id: str,
    action_keys: tuple[str, ...] = ACTION_KEYS,
    image_keys: tuple[str, ...] = IMAGE_KEYS,
    resize_images: bool = True,
    timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    state = [float(observation[key]) for key in action_keys]
    encoded_images: dict[str, str] = {}
    resize_ms = 0.0
    image_encode_ms = 0.0
    base64_ms = 0.0
    for key in image_keys:
        image = _validate_rgb_image(np.asarray(observation[key]))
        if resize_images:
            started = time.perf_counter()
            image = resize_image_for_policy(image, MODEL_IMAGE_SIZE)
            resize_ms += (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        png = encode_png_bytes(image)
        image_encode_ms += (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        encoded_images[key] = base64.b64encode(png).decode("ascii")
        base64_ms += (time.perf_counter() - started) * 1000.0
    if timings is not None:
        timings.update(
            {
                "client_resize_ms": resize_ms,
                "client_image_encode_ms": image_encode_ms,
                "client_base64_ms": base64_ms,
            }
        )
    return {
        "request_id": request_id,
        "task": task,
        "state": state,
        "images": encoded_images,
    }


def request_to_observation(
    request: dict[str, Any],
    action_dim: int = ACTION_DIM,
    image_keys: tuple[str, ...] = IMAGE_KEYS,
    timings: dict[str, float] | None = None,
) -> tuple[dict[str, Any], str, str]:
    request_id = str(request.get("request_id", ""))
    task = request.get("task")
    if not request_id:
        raise ContractError("request_id is required")
    if not isinstance(task, str) or not task.strip():
        raise ContractError("task must be a non-empty string")
    images = request.get("images")
    if not isinstance(images, dict):
        raise ContractError("images must be an object")
    missing = [key for key in image_keys if key not in images]
    if missing:
        raise ContractError(f"missing images: {missing}")
    transport_names = {"third_person": "full_image", "wrist": "wrist_image"}
    unsupported = [key for key in image_keys if key not in transport_names]
    if unsupported:
        raise ContractError(f"unsupported checkpoint camera names: {unsupported}")
    state = validate_state(request.get("state"), action_dim)
    observation = {
        "state": state,
    }
    started = time.perf_counter()
    observation.update({transport_names[key]: decode_png(images[key]) for key in image_keys})
    if timings is not None:
        timings["server_image_decode_ms"] = (time.perf_counter() - started) * 1000.0
    return observation, task.strip(), request_id


def action_vector_to_robot_action(action: Any) -> dict[str, float]:
    vector = np.asarray(action, dtype=np.float32)
    if vector.shape != (ACTION_DIM,) or not np.isfinite(vector).all():
        raise ContractError(f"single action must be finite shape ({ACTION_DIM},), got {vector.shape}")
    return {key: float(value) for key, value in zip(ACTION_KEYS, vector, strict=True)}
