from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

from openvla_pipeline.cli import Option, parse_options
from openvla_pipeline.yaml_config import (
    ConfigDocumentError,
    load_mapping,
    require_exact_keys,
    require_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_PATH = PROJECT_ROOT / "configs/runtime/openvla_piper.yaml"
INSTALLED_CONFIG_PATH = Path(sys.prefix) / "share/openvla-piper/openvla_piper.yaml"
DEFAULT_CONFIG_PATH = SOURCE_CONFIG_PATH


class RuntimeConfigError(ValueError):
    """Raised when the sync deployment configuration is incomplete or invalid."""


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    checkpoint: str | None
    base_model: str | None
    openvla_oft_repo: Path | None
    auth_token_env: str
    max_request_bytes: int

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class ClientConfig:
    model_server: str
    rosbridge_url: str
    piper_repo: Path
    task: str
    max_actions: int
    health_timeout_s: float
    request_timeout_s: float
    inference_log_root: Path
    chunk_diagnostics: bool


@dataclass(frozen=True)
class SafetyConfig:
    allow_live_motion: bool
    gripper_min_m: float
    gripper_max_m: float
    max_arm_step_delta_rad: float


@dataclass(frozen=True)
class RuntimeConfig:
    schema_version: int
    server: ServerConfig
    client: ClientConfig
    safety: SafetyConfig
    source_path: Path


def _expand_path(value: Any, field: str, *, allow_none: bool = False) -> Path | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeConfigError(f"runtime config field {field!r} must be a non-empty path")
    expanded = Path(os.path.expandvars(os.path.expanduser(value.strip())))
    return expanded if expanded.is_absolute() else (PROJECT_ROOT / expanded).resolve()


def _positive_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(f"runtime config field {field!r} must be positive") from exc
    if number <= 0:
        raise RuntimeConfigError(f"runtime config field {field!r} must be positive")
    return number


def _source_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeConfigError(
            f"runtime config field {field!r} must be a local path, hf:// source, or null"
        )
    text = value.strip()
    if text.startswith("hf://"):
        return text
    expanded = Path(os.path.expandvars(os.path.expanduser(text)))
    return str(expanded if expanded.is_absolute() else (PROJECT_ROOT / expanded).resolve())


def load_runtime_config(path: Path | str | None = None) -> RuntimeConfig:
    requested_path = path or os.environ.get("PIPER_OPENVLA_RUNTIME_CONFIG")
    selected = (
        Path(requested_path).expanduser().resolve()
        if requested_path is not None
        else (SOURCE_CONFIG_PATH if SOURCE_CONFIG_PATH.is_file() else INSTALLED_CONFIG_PATH)
    )
    try:
        raw = load_mapping(selected)
        require_exact_keys(
            raw,
            {"schema_version", "server", "client", "safety"},
            where="runtime",
        )
    except ConfigDocumentError as exc:
        raise RuntimeConfigError(str(exc)) from exc
    if raw.get("schema_version") != 1:
        raise RuntimeConfigError(
            f"unsupported runtime config schema_version: {raw.get('schema_version')!r}"
        )

    try:
        server_raw = require_mapping(raw, "server", "runtime")
        client_raw = require_mapping(raw, "client", "runtime")
        safety_raw = require_mapping(raw, "safety", "runtime")
        require_exact_keys(
            server_raw,
            {
                "host", "port", "checkpoint", "base_model", "openvla_oft_repo",
                "auth_token_env", "max_request_bytes",
            },
            where="runtime.server",
        )
        require_exact_keys(
            client_raw,
            {
                "model_server", "rosbridge_url", "piper_repo", "task", "max_actions",
                "health_timeout_s", "request_timeout_s", "inference_log_root",
                "chunk_diagnostics",
            },
            where="runtime.client",
        )
        require_exact_keys(
            safety_raw,
            {"allow_live_motion", "gripper_min_m", "gripper_max_m", "max_arm_step_delta_rad"},
            where="runtime.safety",
        )
    except ConfigDocumentError as exc:
        raise RuntimeConfigError(str(exc)) from exc

    host = server_raw.get("host")
    if not isinstance(host, str) or not host.strip():
        raise RuntimeConfigError("server.host must be a non-empty string")
    try:
        port = int(server_raw.get("port"))
        max_request_bytes = int(server_raw.get("max_request_bytes"))
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError("server.port and server.max_request_bytes must be integers") from exc
    if not 1 <= port <= 65535:
        raise RuntimeConfigError(f"server.port must be 1..65535, got {port}")
    if max_request_bytes <= 0:
        raise RuntimeConfigError("server.max_request_bytes must be positive")

    auth_token_env = server_raw.get("auth_token_env")
    if not isinstance(auth_token_env, str) or not auth_token_env.strip():
        raise RuntimeConfigError("server.auth_token_env must be a non-empty string")

    model_server = client_raw.get("model_server")
    rosbridge_url = client_raw.get("rosbridge_url")
    task = client_raw.get("task")
    if not isinstance(model_server, str) or not model_server.startswith(("http://", "https://")):
        raise RuntimeConfigError("client.model_server must be an HTTP(S) URL")
    if not isinstance(rosbridge_url, str) or not rosbridge_url.startswith(("ws://", "wss://")):
        raise RuntimeConfigError("client.rosbridge_url must be a WebSocket URL")
    if not isinstance(task, str) or not task.strip():
        raise RuntimeConfigError("client.task must be a non-empty string")
    try:
        max_actions = int(client_raw.get("max_actions"))
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError("client.max_actions must be a positive integer") from exc
    if max_actions <= 0:
        raise RuntimeConfigError("client.max_actions must be a positive integer")
    chunk_diagnostics = client_raw.get("chunk_diagnostics")
    if not isinstance(chunk_diagnostics, bool):
        raise RuntimeConfigError("client.chunk_diagnostics must be true or false")

    allow_live_motion = safety_raw.get("allow_live_motion")
    if not isinstance(allow_live_motion, bool):
        raise RuntimeConfigError("safety.allow_live_motion must be true or false")
    try:
        gripper_min_m = float(safety_raw.get("gripper_min_m"))
        gripper_max_m = float(safety_raw.get("gripper_max_m"))
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError("safety gripper bounds must be numbers") from exc
    if gripper_min_m < 0 or gripper_max_m <= gripper_min_m:
        raise RuntimeConfigError(
            "safety gripper range must satisfy 0 <= gripper_min_m < gripper_max_m"
        )

    return RuntimeConfig(
        schema_version=1,
        server=ServerConfig(
            host=host.strip(),
            port=port,
            checkpoint=_source_text(server_raw.get("checkpoint"), "server.checkpoint"),
            base_model=_source_text(server_raw.get("base_model"), "server.base_model"),
            openvla_oft_repo=_expand_path(
                server_raw.get("openvla_oft_repo"),
                "server.openvla_oft_repo",
                allow_none=True,
            ),
            auth_token_env=auth_token_env.strip(),
            max_request_bytes=max_request_bytes,
        ),
        client=ClientConfig(
            model_server=model_server.rstrip("/"),
            rosbridge_url=rosbridge_url,
            piper_repo=_expand_path(client_raw.get("piper_repo"), "client.piper_repo"),
            task=task.strip(),
            max_actions=max_actions,
            health_timeout_s=_positive_number(
                client_raw.get("health_timeout_s"), "client.health_timeout_s"
            ),
            request_timeout_s=_positive_number(
                client_raw.get("request_timeout_s"), "client.request_timeout_s"
            ),
            inference_log_root=_expand_path(
                client_raw.get("inference_log_root"), "client.inference_log_root"
            ),
            chunk_diagnostics=chunk_diagnostics,
        ),
        safety=SafetyConfig(
            allow_live_motion=allow_live_motion,
            gripper_min_m=gripper_min_m,
            gripper_max_m=gripper_max_m,
            max_arm_step_delta_rad=_positive_number(
                safety_raw.get("max_arm_step_delta_rad"),
                "safety.max_arm_step_delta_rad",
            ),
        ),
        source_path=selected,
    )


def _json_ready(config: RuntimeConfig) -> dict[str, Any]:
    payload = asdict(config)

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value

    return convert(payload)


def main(argv: list[str] | None = None) -> None:
    args, _ = parse_options(
        argv,
        (Option("config", converter=Path, help="runtime YAML or legacy JSON path"),),
        description="Validate and print the OpenVLA-Piper runtime config",
    )
    print(json.dumps(_json_ready(load_runtime_config(args.config)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
