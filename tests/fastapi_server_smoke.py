"""CPU-only ASGI contract test for the synchronous FastAPI model server."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from openvla_pipeline.model_io import encode_png
from openvla_pipeline.model_server import create_app


class FakeMergedPolicy:
    action_dim = 7
    image_keys = ("third_person", "wrist")

    def health(self):
        return {"ready": True, "lora_merged": True, "action_shape": [20, 7]}

    def predict(self, observation, task):
        assert observation["state"].shape == (7,)
        assert observation["full_image"].shape == (224, 224, 3)
        assert observation["wrist_image"].shape == (224, 224, 3)
        assert task == "two block pnp"
        return {
            "actions": np.zeros((20, 7), dtype=np.float32).tolist(),
            "action_shape": [20, 7],
            "inference_ms": 1.0,
        }


async def asgi_request(
    app,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    raw_headers = {
        "host": "testserver",
        "content-length": str(len(body)),
        **({"content-type": "application/json"} if payload is not None else {}),
        **(headers or {}),
    }
    messages: list[dict[str, Any]] = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "root_path": "",
            "headers": [
                (key.lower().encode("ascii"), value.encode("ascii"))
                for key, value in raw_headers.items()
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8777),
        },
        receive,
        send,
    )
    status = next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )
    response_body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return status, json.loads(response_body)


async def smoke() -> None:
    policy = FakeMergedPolicy()
    app = create_app(policy, auth_token="test-token", max_request_bytes=8192)

    status, health = await asgi_request(app, "GET", "/health")
    assert status == 200
    assert health["server_framework"] == "fastapi"
    assert health["inference_mode"] == "synchronous"
    assert health["lora_merged"] is True
    status, openapi = await asgi_request(app, "GET", "/openapi.json")
    assert status == 200
    assert set(openapi["paths"]) >= {"/health", "/act"}

    image = encode_png(np.zeros((224, 224, 3), dtype=np.uint8))
    payload = {
        "request_id": "request-1",
        "task": "two block pnp",
        "state": [0.0] * 7,
        "images": {"third_person": image, "wrist": image},
    }
    status, response = await asgi_request(app, "POST", "/act", payload)
    assert status == 401
    assert response == {"error": "invalid bearer token"}

    status, response = await asgi_request(
        app,
        "POST",
        "/act",
        payload,
        headers={"authorization": "Bearer test-token"},
    )
    assert status == 200
    assert response["request_id"] == "request-1"
    assert response["action_shape"] == [20, 7]
    assert response["timings"]["server_request_parse_ms"] >= 0.0
    assert response["timings"]["server_image_decode_ms"] >= 0.0

    invalid = {**payload, "state": [0.0] * 6}
    status, response = await asgi_request(
        app,
        "POST",
        "/act",
        invalid,
        headers={"authorization": "Bearer test-token"},
    )
    assert status == 400
    assert "state must have shape (7,)" in response["error"]

    tiny_app = create_app(policy, auth_token=None, max_request_bytes=8)
    status, response = await asgi_request(tiny_app, "POST", "/act", payload)
    assert status == 400
    assert "Content-Length must be 1..8" in response["error"]

    print("fastapi_routes=/health,/act,/docs,/openapi.json")
    print("fastapi_auth_size_validation=True")
    print("lora_merged_health_contract=True")
    print("fastapi_server_smoke=True")


def main() -> None:
    asyncio.run(smoke())


if __name__ == "__main__":
    main()
