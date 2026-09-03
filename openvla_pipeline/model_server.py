"""Serve the synchronous OpenVLA PiPER policy with FastAPI and Uvicorn."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from openvla_pipeline.config import load_runtime_config
from openvla_pipeline.model_io import ContractError, request_to_observation
from openvla_pipeline.openvla_policy import PiperOpenVLAPolicy


class ActionRequest(BaseModel):
    """JSON wire contract used by the synchronous PiPER inference client."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    task: str
    state: list[float]
    images: dict[str, str]


class RequestSizeLimitMiddleware:
    """Reject oversized `/act` bodies before FastAPI reads or validates them."""

    def __init__(self, app: Any, max_request_bytes: int) -> None:
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == "/act":
            headers = dict(scope.get("headers", ()))
            raw_length = headers.get(b"content-length", b"0")
            try:
                content_length = int(raw_length)
            except ValueError:
                response = JSONResponse(
                    status_code=400, content={"error": "invalid Content-Length"}
                )
                await response(scope, receive, send)
                return
            if content_length <= 0 or content_length > self.max_request_bytes:
                response = JSONResponse(
                    status_code=400,
                    content={
                        "error": (
                            "Content-Length must be "
                            f"1..{self.max_request_bytes}, got {content_length}"
                        )
                    },
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app(
    policy: PiperOpenVLAPolicy,
    *,
    auth_token: str | None,
    max_request_bytes: int,
) -> FastAPI:
    """Wrap one already-loaded, LoRA-merged policy in a FastAPI application."""

    if max_request_bytes <= 0:
        raise ValueError("max_request_bytes must be positive")

    app = FastAPI(
        title="OpenVLA PiPER synchronous inference server",
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.policy = policy
    app.state.auth_token = auth_token
    app.state.max_request_bytes = max_request_bytes
    app.add_middleware(
        RequestSizeLimitMiddleware, max_request_bytes=max_request_bytes
    )

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid request body", "details": exc.errors()},
        )

    @app.get("/health", response_model=None)
    async def health() -> dict[str, Any]:
        return {
            "server_framework": "fastapi",
            "inference_mode": "synchronous",
            **app.state.policy.health(),
        }

    @app.post("/act", response_model=None)
    async def act(
        payload: ActionRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        if app.state.auth_token and authorization != f"Bearer {app.state.auth_token}":
            raise HTTPException(status_code=401, detail="invalid bearer token")

        try:
            observation, task, request_id = request_to_observation(
                payload.model_dump(),
                action_dim=app.state.policy.action_dim,
                image_keys=app.state.policy.image_keys,
            )
            # OpenVLA-OFT inference is intentionally synchronous. Uvicorn runs
            # one worker and PiperOpenVLAPolicy serializes CUDA access.
            result = app.state.policy.predict(observation, task)
        except ContractError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        return {"request_id": request_id, **result}

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args(argv)
    config = load_runtime_config(config_args.config)

    parser = argparse.ArgumentParser(
        description="Serve the LoRA-merged PiPER OpenVLA policy with FastAPI/Uvicorn"
    )
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
        "--host", default=os.environ.get("PIPER_OPENVLA_SERVER_HOST", config.server.host)
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
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default=os.environ.get("PIPER_OPENVLA_LOG_LEVEL", "info"),
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
    app = create_app(
        policy,
        auth_token=os.environ.get(args.auth_token_env) or None,
        max_request_bytes=args.max_request_bytes,
    )
    print(
        json.dumps(
            {
                "event": "server_ready",
                "server_framework": "fastapi",
                "inference_mode": "synchronous",
                **policy.health(),
            },
            indent=2,
        ),
        flush=True,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        log_level=args.log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
