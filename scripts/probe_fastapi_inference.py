#!/usr/bin/env python3
"""Send one deterministic, ROS-free request through the real FastAPI policy."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from openvla_pipeline.cli import Option, parse_options, selected_option, usage_error
from openvla_pipeline.config import load_runtime_config
from openvla_pipeline.model_io import ACTION_KEYS, observation_to_request, validate_action_chunk
from openvla_pipeline.piper_dry_run import request_json


def main() -> None:
    config = load_runtime_config(selected_option(None, "config", Path))
    args, _ = parse_options(
        None,
        (
            Option("config", converter=Path, default=config.source_path),
            Option("server", default=config.client.model_server),
            Option("checkpoint", converter=Path, default=Path(config.server.checkpoint) if config.server.checkpoint else None),
            Option("task", default=config.client.task),
            Option("timeout_s", converter=float, default=120.0),
            Option("token_env", default=config.server.auth_token_env),
            Option("image_size", converter=int, default=256),
        ),
        description="Run one synthetic image-pair inference through FastAPI (no ROS output)",
    )
    if args.checkpoint is None:
        usage_error("--checkpoint is required")
    if args.timeout_s <= 0 or args.image_size < 16:
        usage_error("--timeout-s must be positive and --image-size must be at least 16")

    checkpoint = args.checkpoint.expanduser().resolve()
    metadata = json.loads((checkpoint / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    contract = metadata["training_contract"]
    robot_contract = contract["robot_contract"]
    camera_names = tuple(robot_contract["camera_names"])
    statistics = json.loads((checkpoint / "dataset_statistics.json").read_text(encoding="utf-8"))
    state = statistics[robot_contract["robot_type"]]["proprio"]["mean"]
    image = np.full((args.image_size, args.image_size, 3), 127, dtype=np.uint8)
    observation = {
        **dict(zip(ACTION_KEYS, state, strict=True)),
        **{camera: image for camera in camera_names},
    }
    request_id = f"probe-{uuid.uuid4().hex}"
    payload = observation_to_request(
        observation,
        args.task,
        request_id,
        action_keys=ACTION_KEYS,
        image_keys=camera_names,
    )
    response = request_json(
        f"{args.server.rstrip('/')}/act",
        payload,
        os.environ.get(args.token_env) or None,
        args.timeout_s,
    )
    if response.get("request_id") != request_id:
        raise RuntimeError("FastAPI response request_id mismatch")
    actions = validate_action_chunk(
        response.get("actions"),
        (int(contract["num_actions_chunk"]), int(contract["action_dim"])),
    )
    print(
        json.dumps(
            {
                "fastapi_inference": True,
                "request_id": request_id,
                "action_shape": list(actions.shape),
                "first_action": actions[0].tolist(),
                "inference_ms": response.get("inference_ms"),
                "peak_vram_mib": response.get("peak_vram_mib"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
