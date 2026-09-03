"""Resolve one launch contract for the sync and async inference wrappers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import sys
from typing import Literal
from urllib.parse import urlsplit

from openvla_pipeline.cli import selected_option, usage_error, without_switch
from openvla_pipeline.config import RuntimeConfigError, load_runtime_config


LaunchKind = Literal["sync", "async"]
LOCAL_SERVER_HOSTS = {"127.0.0.1", "localhost"}


@dataclass(frozen=True)
class LaunchPlan:
    kind: LaunchKind
    config_source: Path
    server_url: str
    checkpoint: str | None
    server_host: str
    server_port: int
    auto_start_local_server: bool
    action_chunk: int | None
    auth_token_env: str
    health_timeout_s: float
    health_attempts: int


def _selected_config_path(argv: list[str]) -> Path | None:
    return selected_option(argv, "config", Path)


def _parse_client_settings(kind: LaunchKind, argv: list[str]):
    if kind == "sync":
        from openvla_pipeline.ros_node import parse_settings
    else:
        from openvla_async_pipeline.ros_node import parse_settings
    return parse_settings(argv)


def _checkpoint_action_chunk(checkpoint: str | None) -> int | None:
    if checkpoint is None:
        return None
    if checkpoint.startswith("hf://"):
        return None
    metadata_path = Path(checkpoint) / "checkpoint_metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        value = metadata["training_contract"]["num_actions_chunk"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeConfigError(
            f"cannot read checkpoint action chunk from {metadata_path}: {exc}"
        ) from exc
    if not isinstance(value, int) or value <= 0:
        raise RuntimeConfigError(
            f"checkpoint num_actions_chunk must be positive, got {value!r}"
        )
    return value


def _expected_action_chunk(action_chunk: int | None) -> int | None:
    raw_value = os.environ.get("PIPER_ACTION_CHUNK")
    if raw_value is None:
        return action_chunk
    try:
        expected = int(raw_value)
    except ValueError as exc:
        raise RuntimeConfigError(
            f"PIPER_ACTION_CHUNK must be a positive integer, got {raw_value!r}"
        ) from exc
    if expected <= 0:
        raise RuntimeConfigError(
            f"PIPER_ACTION_CHUNK must be a positive integer, got {raw_value!r}"
        )
    if action_chunk is not None and expected != action_chunk:
        raise RuntimeConfigError(
            "PIPER_ACTION_CHUNK conflicts with checkpoint metadata: "
            f"environment={expected} checkpoint={action_chunk}"
        )
    return action_chunk if action_chunk is not None else expected


def resolve_launch_plan(kind: LaunchKind, argv: list[str]) -> LaunchPlan:
    config = load_runtime_config(_selected_config_path(argv))
    settings = _parse_client_settings(kind, argv)
    checkpoint_text = os.environ.get("PIPER_OPENVLA_CHECKPOINT")
    checkpoint = checkpoint_text or config.server.checkpoint
    if checkpoint is not None and not checkpoint.startswith("hf://"):
        checkpoint = str(Path(checkpoint).expanduser().resolve())

    parsed_url = urlsplit(settings.model_server)
    if parsed_url.scheme not in {"http", "https"} or parsed_url.hostname is None:
        raise RuntimeConfigError(
            f"model server must be an absolute HTTP(S) URL, got {settings.model_server!r}"
        )
    if parsed_url.username is not None or parsed_url.password is not None:
        raise RuntimeConfigError("model server URL must not contain credentials")
    if parsed_url.path not in {"", "/"} or parsed_url.query or parsed_url.fragment:
        raise RuntimeConfigError(
            "model server URL must not contain a path, query, or fragment: "
            f"{settings.model_server!r}"
        )
    try:
        server_port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
    except ValueError as exc:
        raise RuntimeConfigError(f"invalid model server port: {settings.model_server!r}") from exc

    action_chunk = _expected_action_chunk(_checkpoint_action_chunk(checkpoint))
    health_timeout_s = float(settings.health_timeout_s)
    return LaunchPlan(
        kind=kind,
        config_source=config.source_path,
        server_url=settings.model_server.rstrip("/"),
        checkpoint=checkpoint,
        server_host=parsed_url.hostname,
        server_port=server_port,
        auto_start_local_server=(
            parsed_url.scheme == "http" and parsed_url.hostname in LOCAL_SERVER_HOSTS
        ),
        action_chunk=action_chunk,
        auth_token_env=config.server.auth_token_env,
        health_timeout_s=health_timeout_s,
        health_attempts=max(1, math.ceil(health_timeout_s)),
    )


def _line_values(plan: LaunchPlan) -> tuple[str, ...]:
    return (
        str(plan.config_source),
        plan.server_url,
        "" if plan.checkpoint is None else str(plan.checkpoint),
        plan.server_host,
        str(plan.server_port),
        "1" if plan.auto_start_local_server else "0",
        "" if plan.action_chunk is None else str(plan.action_chunk),
        plan.auth_token_env,
        str(plan.health_attempts),
    )


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: python -m openvla_pipeline.launch_plan {sync|async} [--lines] [client options]")
        return
    kind = args.pop(0)
    if kind not in {"sync", "async"}:
        usage_error(f"runtime kind must be sync or async, got {kind!r}")
    lines, forwarded = without_switch(args, "lines")
    plan = resolve_launch_plan(kind, forwarded)  # type: ignore[arg-type]
    if lines:
        print("\n".join(_line_values(plan)))
    else:
        print(json.dumps(asdict(plan), indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
