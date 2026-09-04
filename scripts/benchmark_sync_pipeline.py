#!/usr/bin/env python3
"""Benchmark the synchronous FastAPI path before/after client-side 224 resize."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import uuid
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from openvla_pipeline.config import load_runtime_config
from openvla_pipeline.cli import Option, parse_options, selected_option, usage_error
from openvla_pipeline.model_io import (
    ACTION_KEYS,
    CLIENT_RESIZE_ALGORITHM,
    observation_to_request,
    validate_action_chunk,
)
from openvla_pipeline.piper_dry_run import request_json, request_json_with_timings


TIMING_FIELDS = (
    "observation_acquire_ms",
    "client_resize_ms",
    "client_image_encode_ms",
    "client_base64_ms",
    "client_json_serialize_ms",
    "http_roundtrip_ms",
    "server_request_parse_ms",
    "server_image_decode_ms",
    "server_preprocess_ms",
    "model_forward_ms",
    "server_response_serialize_ms",
    "total_client_wall_ms",
)


def _jpeg_fixture(seed: int) -> str:
    rng = np.random.default_rng(seed)
    height, width = 480, 640
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack(
        (
            (xx + rng.integers(0, 32, size=(height, width))) % 256,
            (yy + rng.integers(0, 32, size=(height, width))) % 256,
            ((xx // 2 + yy // 2) + rng.integers(0, 32, size=(height, width))) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("failed to create benchmark JPEG fixture")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _acquire_observation(
    encoded_images: dict[str, str],
    state: list[float],
) -> dict[str, Any]:
    observation: dict[str, Any] = dict(zip(ACTION_KEYS, state, strict=True))
    for camera, encoded in encoded_images.items():
        jpeg = base64.b64decode(encoded)
        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to decode benchmark fixture {camera}")
        observation[camera] = np.ascontiguousarray(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8
        )
    return observation


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"samples": len(records)}
    for field in (*TIMING_FIELDS, "request_bytes", "response_bytes"):
        values = np.asarray([record[field] for record in records], dtype=np.float64)
        summary[field] = {
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
        }
    return summary


def _run_mode(
    *,
    mode: str,
    resize_images: bool,
    server: str,
    task: str,
    token: str | None,
    timeout_s: float,
    warmup: int,
    iterations: int,
    state: list[float],
    camera_names: tuple[str, ...],
    action_shape: tuple[int, int],
    encoded_images: dict[str, str],
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    records: list[dict[str, Any]] = []
    outputs: list[np.ndarray] = []
    total_runs = warmup + iterations
    for index in range(total_runs):
        total_started = time.perf_counter()
        acquire_started = time.perf_counter()
        observation = _acquire_observation(encoded_images, state)
        timings: dict[str, float | int] = {
            "observation_acquire_ms": (time.perf_counter() - acquire_started) * 1000.0
        }
        request_id = f"benchmark-{mode}-{index}-{uuid.uuid4().hex}"
        payload = observation_to_request(
            observation,
            task,
            request_id,
            action_keys=ACTION_KEYS,
            image_keys=camera_names,
            resize_images=resize_images,
            timings=timings,
        )
        response, transport_timings = request_json_with_timings(
            f"{server}/act", payload, token, timeout_s
        )
        timings.update(transport_timings)
        timings.update(response.get("timings", {}))
        timings["total_client_wall_ms"] = (time.perf_counter() - total_started) * 1000.0
        missing = [field for field in TIMING_FIELDS if field not in timings]
        if missing:
            raise RuntimeError(f"server/client timing contract missing fields: {missing}")
        actions = validate_action_chunk(response.get("actions"), action_shape).copy()
        if index >= warmup:
            records.append(
                {
                    "index": index - warmup,
                    **{field: float(timings[field]) for field in TIMING_FIELDS},
                    "request_bytes": int(timings["request_bytes"]),
                    "response_bytes": int(timings["response_bytes"]),
                }
            )
            outputs.append(actions)
        print(
            f"mode={mode} run={index + 1}/{total_runs} "
            f"warmup={index < warmup} wall_ms={timings['total_client_wall_ms']:.2f}",
            flush=True,
        )
    return records, outputs


def main() -> None:
    config = load_runtime_config(selected_option(None, "config", Path))
    args, _ = parse_options(
        None,
        (
            Option("config", converter=Path, default=config.source_path),
            Option("server", default=config.client.model_server),
            Option("task", default=config.client.task),
            Option("iterations", converter=int, default=50),
            Option("warmup", converter=int, default=5),
            Option("timeout_s", converter=float, default=120.0),
            Option("token_env", default=config.server.auth_token_env),
            Option("output", converter=Path, default=None),
        ),
        description="Benchmark sync requests before/after official client 224 resize",
    )
    if args.iterations < 1 or args.warmup < 0 or args.timeout_s <= 0:
        usage_error("iterations must be positive, warmup non-negative, and timeout-s positive")

    server = args.server.rstrip("/")
    token = os.environ.get(args.token_env) or None
    health = request_json(f"{server}/health", None, token, args.timeout_s)
    if health.get("ready") is not True:
        raise RuntimeError(f"policy server is not ready: {health}")
    checkpoint = Path(health["checkpoint"])
    metadata = json.loads((checkpoint / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    contract = metadata["training_contract"]
    robot_contract = contract["robot_contract"]
    statistics = json.loads((checkpoint / "dataset_statistics.json").read_text(encoding="utf-8"))
    state = list(statistics[robot_contract["robot_type"]]["proprio"]["mean"])
    camera_names = tuple(robot_contract["camera_names"])
    action_shape = (int(contract["num_actions_chunk"]), int(contract["action_dim"]))
    encoded_images = {
        camera: _jpeg_fixture(20260904 + index)
        for index, camera in enumerate(camera_names)
    }

    before_records, before_outputs = _run_mode(
        mode="before_full_resolution",
        resize_images=False,
        server=server,
        task=args.task,
        token=token,
        timeout_s=args.timeout_s,
        warmup=args.warmup,
        iterations=args.iterations,
        state=state,
        camera_names=camera_names,
        action_shape=action_shape,
        encoded_images=encoded_images,
    )
    after_records, after_outputs = _run_mode(
        mode="after_official_224",
        resize_images=True,
        server=server,
        task=args.task,
        token=token,
        timeout_s=args.timeout_s,
        warmup=args.warmup,
        iterations=args.iterations,
        state=state,
        camera_names=camera_names,
        action_shape=action_shape,
        encoded_images=encoded_images,
    )
    paired_action_delta = max(
        float(np.max(np.abs(before - after)))
        for before, after in zip(before_outputs, after_outputs, strict=True)
    )
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or PROJECT_ROOT / "artifacts" / "benchmarks" / f"sync_224_{created_at}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "created_at": created_at,
        "benchmark_contract": {
            "server": server,
            "checkpoint": str(checkpoint),
            "iterations": args.iterations,
            "warmup_per_mode": args.warmup,
            "observation_source": "deterministic ROS CompressedImage-compatible base64 JPEG fixture",
            "image_source_shape": [480, 640, 3],
            "client_resize_algorithm": CLIENT_RESIZE_ALGORITHM,
            "camera_names": list(camera_names),
            "action_shape": list(action_shape),
        },
        "before": {"summary": _summary(before_records), "records": before_records},
        "after": {"summary": _summary(after_records), "records": after_records},
        "max_paired_action_abs_delta": paired_action_delta,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "benchmark_contract": report["benchmark_contract"],
                "before": report["before"]["summary"],
                "after": report["after"]["summary"],
                "max_paired_action_abs_delta": paired_action_delta,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
