"""Serve the OpenVLA Piper policy over a local HTTP API."""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from openvla_pipeline.model_io import ContractError, request_to_observation
from openvla_pipeline.openvla_policy import PiperOpenVLAPolicy
from openvla_pipeline.config import load_runtime_config



class OpenVLAModelServer(ThreadingHTTPServer):
    policy: PiperOpenVLAPolicy
    auth_token: str | None
    max_request_bytes: int


class OpenVLARequestHandler(BaseHTTPRequestHandler):
    server: OpenVLAModelServer

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._send(HTTPStatus.OK, self.server.policy.health())

    def do_POST(self) -> None:
        if self.path != "/act":
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if self.server.auth_token and self.headers.get("Authorization") != f"Bearer {self.server.auth_token}":
            self._send(HTTPStatus.UNAUTHORIZED, {"error": "invalid bearer token"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > self.server.max_request_bytes:
                raise ContractError(
                    "Content-Length must be "
                    f"1..{self.server.max_request_bytes}, got {content_length}"
                )
            request = json.loads(self.rfile.read(content_length))
            observation, task, request_id = request_to_observation(
                request,
                action_dim=self.server.policy.action_dim,
                image_keys=self.server.policy.image_keys,
            )
            result = self.server.policy.predict(observation, task)
            self._send(HTTPStatus.OK, {"request_id": request_id, **result})
        except (ContractError, json.JSONDecodeError) as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # keep the control client from hanging on model errors
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"client={self.client_address[0]} {format_string % args}", flush=True)

    def _send(self, status: HTTPStatus, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args(argv)
    config = load_runtime_config(config_args.config)

    parser = argparse.ArgumentParser(description="Serve the final Piper OpenVLA policy over localhost HTTP")
    parser.add_argument("--config", type=Path, default=config.source_path)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            os.environ["PIPER_OPENVLA_CHECKPOINT"]
            if os.environ.get("PIPER_OPENVLA_CHECKPOINT")
            else config.server.checkpoint
        ),
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=(
            os.environ["OPENVLA_BASE_MODEL"]
            if os.environ.get("OPENVLA_BASE_MODEL")
            else config.server.base_model
        ),
    )
    parser.add_argument(
        "--openvla-oft-repo",
        type=Path,
        default=(
            Path(os.environ["OPENVLA_OFT_REPO"])
            if os.environ.get("OPENVLA_OFT_REPO")
            else config.server.openvla_oft_repo
        ),
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("PIPER_OPENVLA_SERVER_HOST", config.server.host),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PIPER_OPENVLA_SERVER_PORT", config.server.port)),
    )
    parser.add_argument(
        "--auth-token-env",
        default=os.environ.get(
            "PIPER_OPENVLA_AUTH_TOKEN_ENV", config.server.auth_token_env
        ),
    )
    parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=int(
            os.environ.get(
                "PIPER_OPENVLA_MAX_REQUEST_BYTES", config.server.max_request_bytes
            )
        ),
    )
    args = parser.parse_args(argv)
    if args.checkpoint is None:
        parser.error(
            "--checkpoint is required (or set server.checkpoint / PIPER_OPENVLA_CHECKPOINT)"
        )
    if args.max_request_bytes <= 0:
        parser.error("--max-request-bytes must be positive")
    return args


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.config)
    policy = PiperOpenVLAPolicy(
        args.checkpoint,
        args.base_model,
        args.openvla_oft_repo,
        config.safety.max_arm_step_delta_rad,
    )
    model_server = OpenVLAModelServer((args.host, args.port), OpenVLARequestHandler)
    model_server.policy = policy
    model_server.auth_token = os.environ.get(args.auth_token_env) or None
    model_server.max_request_bytes = args.max_request_bytes
    print(json.dumps({"event": "server_ready", **policy.health()}, indent=2), flush=True)
    try:
        model_server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        model_server.server_close()


# Compatibility aliases for pre-refactor imports.
PolicyHTTPServer = OpenVLAModelServer
PolicyHandler = OpenVLARequestHandler


if __name__ == "__main__":
    main()
