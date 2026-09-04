"""Serve the synchronous OpenVLA PiPER policy with FastAPI and Uvicorn."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from openvla_pipeline.cli import Option, parse_options, selected_option, usage_error
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


class RequestTimingMiddleware:
    """Expose request parsing and response rendering boundaries without profiling."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/act":
            await self.app(scope, receive, send)
            return

        state = scope.setdefault("state", {})

        async def timed_receive() -> dict[str, Any]:
            message = await receive()
            if message.get("type") == "http.request" and not message.get("more_body", False):
                state["request_body_received_ns"] = time.perf_counter_ns()
            return message

        async def timed_send(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                started_ns = state.get("response_serialize_started_ns")
                if started_ns is not None:
                    duration_ms = (time.perf_counter_ns() - int(started_ns)) / 1_000_000.0
                    headers = list(message.get("headers", ()))
                    headers.append(
                        (b"server-timing", f"response_serialize;dur={duration_ms:.6f}".encode("ascii"))
                    )
                    message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, timed_receive, timed_send)


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
    app.add_middleware(RequestTimingMiddleware)

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
        request: Request,
        payload: ActionRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        endpoint_started_ns = time.perf_counter_ns()
        if app.state.auth_token and authorization != f"Bearer {app.state.auth_token}":
            raise HTTPException(status_code=401, detail="invalid bearer token")

        server_timings: dict[str, float] = {}
        body_received_ns = getattr(request.state, "request_body_received_ns", None)
        if body_received_ns is not None:
            server_timings["server_request_parse_ms"] = (
                endpoint_started_ns - int(body_received_ns)
            ) / 1_000_000.0
        try:
            observation, task, request_id = request_to_observation(
                payload.model_dump(),
                action_dim=app.state.policy.action_dim,
                image_keys=app.state.policy.image_keys,
                timings=server_timings,
            )
            # OpenVLA-OFT inference is intentionally synchronous. Uvicorn runs
            # one worker and PiperOpenVLAPolicy serializes CUDA access.
            result = app.state.policy.predict(observation, task)
            server_timings.update(result.pop("server_timings", {}))
        except ContractError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        response = {"request_id": request_id, **result, "timings": server_timings}
        request.state.response_serialize_started_ns = time.perf_counter_ns()
        return response

    return app


def parse_args(argv: list[str] | None = None):
    config = load_runtime_config(selected_option(argv, "config", Path))
    args, _ = parse_options(
        argv,
        (
            Option("config", converter=Path, default=config.source_path),
            Option(
                "checkpoint",
                default=os.environ.get("PIPER_OPENVLA_CHECKPOINT") or config.server.checkpoint,
            ),
            Option(
                "base_model",
                default=os.environ.get("OPENVLA_BASE_MODEL") or config.server.base_model,
            ),
            Option(
                "openvla_oft_repo",
                converter=Path,
                default=(Path(os.environ["OPENVLA_OFT_REPO"]) if os.environ.get("OPENVLA_OFT_REPO") else config.server.openvla_oft_repo),
            ),
            Option("host", default=os.environ.get("PIPER_OPENVLA_SERVER_HOST", config.server.host)),
            Option("port", converter=int, default=int(os.environ.get("PIPER_OPENVLA_SERVER_PORT", config.server.port))),
            Option("auth_token_env", default=os.environ.get("PIPER_OPENVLA_AUTH_TOKEN_ENV", config.server.auth_token_env)),
            Option("max_request_bytes", converter=int, default=int(os.environ.get("PIPER_OPENVLA_MAX_REQUEST_BYTES", config.server.max_request_bytes))),
            Option(
                "log_level",
                choices=("critical", "error", "warning", "info", "debug", "trace"),
                default=os.environ.get("PIPER_OPENVLA_LOG_LEVEL", "info"),
            ),
        ),
        description="Serve the LoRA-merged PiPER OpenVLA policy with FastAPI/Uvicorn",
    )
    if args.checkpoint is None:
        usage_error(
            "--checkpoint is required (or set server.checkpoint / PIPER_OPENVLA_CHECKPOINT)"
        )
    if args.max_request_bytes <= 0:
        usage_error("--max-request-bytes must be positive")
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
