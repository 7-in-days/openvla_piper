"""Run the low-level Piper observation and OpenVLA action dry-run client."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from openvla_pipeline.model_io import (
    LIVE_CONFIRMATION,
    ContractError,
    observation_to_request,
    validate_action_chunk,
)
from openvla_pipeline.config import load_runtime_config
from openvla_pipeline.piper_runtime import PiperModelContract


def request_json(url: str, payload: dict[str, Any] | None, token: str | None, timeout: float) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=body, headers=headers, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"policy server HTTP {exc.code}: {detail}") from exc
    if "error" in result:
        raise RuntimeError(f"policy server error: {result['error']}")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args(argv)
    config = load_runtime_config(config_args.config)

    parser = argparse.ArgumentParser(description="Run the Piper bridge against the local OpenVLA policy server")
    parser.add_argument("--config", type=Path, default=config.source_path)
    parser.add_argument("--server", default=config.client.model_server)
    parser.add_argument("--task", default=config.client.task)
    parser.add_argument("--max-actions", type=int, default=config.client.max_actions)
    parser.add_argument("--health-timeout-s", type=float, default=config.client.health_timeout_s)
    parser.add_argument("--request-timeout-s", type=float, default=config.client.request_timeout_s)
    parser.add_argument("--rosbridge-url", default=config.client.rosbridge_url)
    parser.add_argument("--piper-repo", type=Path, default=config.client.piper_repo)
    parser.add_argument("--live", action="store_true", help="publish actions; requires confirmation env var")
    parser.add_argument("--token-env", default=config.server.auth_token_env)
    args = parser.parse_args(argv)
    if args.max_actions < 1:
        parser.error("--max-actions must be positive")
    if args.health_timeout_s <= 0 or args.request_timeout_s <= 0:
        parser.error("request timeouts must be positive")
    return args


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.config)
    args.server = args.server.rstrip("/")
    if args.live and not config.safety.allow_live_motion:
        raise RuntimeError(
            "live mode is disabled by safety.allow_live_motion in the runtime config"
        )
    if args.live and os.environ.get(LIVE_CONFIRMATION) != "YES":
        raise RuntimeError(f"live mode requires {LIVE_CONFIRMATION}=YES and --live")

    piper_repo = args.piper_repo.resolve()
    sys.path.insert(0, str(piper_repo))
    from piper_bridge.config_piper_bridge import PiperBridgeRobotConfig
    from piper_bridge.piper_bridge_robot import PiperBridgeRobot

    token = os.environ.get(args.token_env) or None
    model_server_status = request_json(
        f"{args.server}/health",
        None,
        token,
        args.health_timeout_s,
    )
    if model_server_status.get("ready") is not True:
        raise ContractError(
            f"policy server health contract mismatch: {model_server_status}"
        )

    robot_config = PiperBridgeRobotConfig(rosbridge_url=args.rosbridge_url)
    piper_model_contract = PiperModelContract.from_runtime(
        model_server_status,
        robot_config,
    )
    piper_robot = PiperBridgeRobot(robot_config)
    sent = 0
    try:
        piper_robot.connect()
        mode = "live" if args.live else "dry-run"
        print(
            json.dumps(
                {
                    "event": "client_ready",
                    "mode": mode,
                    "runtime": json.loads(
                        piper_model_contract.describe(
                            robot_config.frame_topic,
                            robot_config.output_topic,
                        )
                    ),
                    "server": model_server_status,
                },
                indent=2,
            ),
            flush=True,
        )

        while sent < args.max_actions:
            observation_stream_status = piper_robot.get_stream_status()
            if not observation_stream_status.get("ready"):
                raise RuntimeError(
                    "Piper observation stream is not ready: "
                    f"{observation_stream_status}"
                )
            observation = piper_robot.get_observation()
            request_id = uuid.uuid4().hex
            request = observation_to_request(
                observation,
                args.task,
                request_id,
                action_keys=piper_model_contract.action_keys,
                image_keys=piper_model_contract.camera_names,
            )
            response = request_json(f"{args.server}/act", request, token, args.request_timeout_s)
            if response.get("request_id") != request_id:
                raise ContractError("policy response request_id mismatch")
            actions = validate_action_chunk(
                response.get("actions"),
                piper_model_contract.action_shape,
            )

            for action in actions:
                if sent >= args.max_actions:
                    break
                tick_started = time.perf_counter()
                robot_action = {
                    key: float(value)
                    for key, value in zip(
                        piper_model_contract.action_keys,
                        action,
                        strict=True,
                    )
                }
                if args.live:
                    piper_robot.send_action(robot_action)
                print(
                    json.dumps(
                        {
                            "event": "action",
                            "mode": mode,
                            "index": sent,
                            "request_id": request_id,
                            "values": [round(float(value), 6) for value in action],
                            "server_inference_ms": round(float(response["inference_ms"]), 2),
                        }
                    ),
                    flush=True,
                )
                sent += 1
                time.sleep(
                    max(
                        0.0,
                        piper_model_contract.control_interval_s
                        - (time.perf_counter() - tick_started),
                    )
                )
    finally:
        if piper_robot.is_connected:
            piper_robot.disconnect()
        print(json.dumps({"event": "client_stopped", "actions_processed": sent}), flush=True)


if __name__ == "__main__":
    main()
