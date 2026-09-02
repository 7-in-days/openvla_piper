"""Write atomic, replayable OpenVLA inference-session logs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import threading
import time
from typing import Any
import uuid


SCHEMA_VERSION = 3
OBSERVABILITY_SCHEMA_VERSION = 1
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
MAX_QUEUE_SIZE = 8192


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        view = memoryview(encoded)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _queue_size(configured: int | None) -> int:
    raw = (
        str(configured)
        if configured is not None
        else os.environ.get("PIPER_OPENVLA_LOG_QUEUE_SIZE", "1024")
    )
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("PIPER_OPENVLA_LOG_QUEUE_SIZE must be an integer") from exc
    if not 1 <= value <= MAX_QUEUE_SIZE:
        raise ValueError(
            f"inference log queue size must be 1..{MAX_QUEUE_SIZE}, got {value}"
        )
    return value


def new_session_id(configured: str | None = None) -> str:
    if configured:
        if not SESSION_ID_PATTERN.fullmatch(configured):
            raise ValueError("session id must use letters, numbers, dot, underscore, or hyphen")
        return configured
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Keep the directory contract used by vla-piper's inference_session_logger.
    return f"session_{now}_{uuid.uuid4().hex[:8]}"


class InferenceSessionLogger:
    """Non-blocking local JSON logger compatible with vla-piper's session layout.

    Metadata stays out of ROS: vla-piper's ROS contract contains only the SyncedFrame
    input and JointState output.  Disk writes happen on a bounded worker queue so the
    20 Hz action loop never waits for fsync.
    """

    def __init__(
        self,
        root: Path,
        session_id: str | None,
        config: dict[str, Any],
        queue_size: int | None = None,
        core_enabled: bool = True,
    ) -> None:
        self.session_id = new_session_id(session_id)
        self.session_dir = root.expanduser().resolve() / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.config_path = self.session_dir / "inference_config.json"
        self.log_path = self.session_dir / "inference_log.jsonl"
        self.observability_path = self.session_dir / "inference_observability.jsonl"
        self._descriptor = os.open(
            self.log_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
        )
        self._observability_descriptor = os.open(
            self.observability_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o644,
        )
        self.core_enabled = bool(core_enabled)
        self.queue_size = _queue_size(queue_size)
        self._queue: queue.Queue[tuple[str, bytes] | None] = queue.Queue(
            maxsize=self.queue_size
        )
        self._error: str | None = None
        self._dropped = 0
        self._closed = False
        self._counts: dict[str, int] = {}
        self._config = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "created_at_ns": time.time_ns(),
            "clock_contract": {
                "request_response_publish": "inference_pc_wall_time_ns",
                "observation_source_stamps": "ROS_message_header_source_clock",
                "server_model_inference_ms": "server_reported_duration",
                "cross_machine_subtraction_requires_clock_sync": True,
            },
            "files": {
                "inference_config": self.config_path.name,
                "inference_log": self.log_path.name,
                "inference_observability": self.observability_path.name,
            },
            "logging": {
                "core_enabled": self.core_enabled,
                "observability_enabled": True,
                "queue_size": self.queue_size,
                "max_queue_size": MAX_QUEUE_SIZE,
                "writer_threads": 1,
            },
            **_json_safe(config),
        }
        _atomic_json(self.config_path, self._config)
        self._thread = threading.Thread(
            target=self._writer,
            name="piper_openvla_inference_jsonl",
            daemon=True,
        )
        self._thread.start()
        server_health = self._config.get("server_health", {})
        self.append_observability(
            {
                "event": "session_meta",
                "schema_version": OBSERVABILITY_SCHEMA_VERSION,
                "session_id": self.session_id,
                "created_at_ns": self._config["created_at_ns"],
                "model_identity": {
                    "checkpoint": server_health.get("checkpoint"),
                    "checkpoint_step": server_health.get("checkpoint_step"),
                    "base_model": server_health.get("base_model"),
                },
                "resolved_oft_contract": server_health.get(
                    "resolved_oft_contract"
                ),
                "clock_contract": self._config["clock_contract"],
            }
        )

    def append_chunk(
        self,
        *,
        chunk_id: int,
        request_stamp_ns: int,
        observation_packet: dict[str, Any],
        instruction: str,
        publish_stamp_ns: int,
    ) -> None:
        """Write the exact seven-field vla-piper chunk-row contract.

        One successful `/piper/inference/chunk` publication produces one JSONL
        row.  Dry-run action ticks are deliberately not separate JSONL rows.
        """
        if not self.core_enabled or self._closed or self._error is not None:
            return
        record = {
            "chunk_id": int(chunk_id),
            "request_stamp_ns": int(request_stamp_ns),
            "obs_third_person_stamp_ns": observation_packet.get(
                "obs_third_person_stamp_ns"
            ),
            "obs_wrist_stamp_ns": observation_packet.get("obs_wrist_stamp_ns"),
            "obs_qmeas_stamp_ns": observation_packet.get("obs_qmeas_stamp_ns"),
            "instruction": str(instruction),
            "publish_stamp_ns": int(publish_stamp_ns),
        }
        encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        try:
            self._queue.put_nowait(("core", encoded))
            self._counts["core_chunk"] = self._counts.get("core_chunk", 0) + 1
        except queue.Full:
            self._dropped += 1
            self._error = "JSONL writer queue full"
            raise RuntimeError(self._error)

    def append_observability(self, record: dict[str, Any]) -> None:
        """Write non-core request/timing metadata without changing the seven fields."""
        if self._closed or self._error is not None:
            return
        encoded = (
            json.dumps(_json_safe(record), ensure_ascii=False, separators=(",", ":"))
            + "\n"
        ).encode()
        try:
            self._queue.put_nowait(("observability", encoded))
            event = str(record.get("event", "unknown"))
            count_key = f"observability_{event}"
            self._counts[count_key] = self._counts.get(count_key, 0) + 1
        except queue.Full:
            self._dropped += 1
            self._error = "JSONL writer queue full"
            raise RuntimeError(self._error)

    def _writer(self) -> None:
        try:
            while True:
                first = self._queue.get()
                if first is None:
                    break
                batch = [first]
                stopping = False
                while len(batch) < 128:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        stopping = True
                        break
                    batch.append(item)
                touched_descriptors: set[int] = set()
                for destination, encoded in batch:
                    descriptor = (
                        self._descriptor
                        if destination == "core"
                        else self._observability_descriptor
                    )
                    view = memoryview(encoded)
                    while view:
                        view = view[os.write(descriptor, view) :]
                    touched_descriptors.add(descriptor)
                for descriptor in touched_descriptors:
                    os.fsync(descriptor)
                if stopping:
                    break
        except Exception as exc:  # noqa: BLE001 -- surfaced in the final session config
            self._error = f"{type(exc).__name__}: {exc}"

    def close(self, termination_reason: str, actions: int, chunks: int) -> None:
        if self._closed:
            return
        try:
            self.append_observability(
                {
                    "event": "session_end",
                    "schema_version": OBSERVABILITY_SCHEMA_VERSION,
                    "session_id": self.session_id,
                    "ended_at_ns": time.time_ns(),
                    "counts": {
                        "actions": int(actions),
                        "chunks": int(chunks),
                        **self._counts,
                    },
                    "termination_reason": str(termination_reason),
                }
            )
        except RuntimeError:
            pass
        self._closed = True
        try:
            self._queue.put(None, timeout=2)
        except queue.Full:
            self._error = self._error or "JSONL writer close queue timeout"
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            self._error = self._error or "JSONL writer did not stop"
        try:
            os.fsync(self._descriptor)
        finally:
            os.close(self._descriptor)
        try:
            os.fsync(self._observability_descriptor)
        finally:
            os.close(self._observability_descriptor)
        artifacts = {
            "inference_config": {
                "enabled": True,
                "path": self.config_path.name,
                "exists": self.config_path.is_file(),
            },
            "inference_log": {
                "enabled": self.core_enabled,
                "path": self.log_path.name,
                "exists": self.log_path.is_file(),
                "size_bytes": self.log_path.stat().st_size,
                "writer_dropped": self._dropped,
                "writer_error": self._error,
            },
            "inference_observability": {
                "enabled": True,
                "path": self.observability_path.name,
                "exists": self.observability_path.is_file(),
                "size_bytes": self.observability_path.stat().st_size,
                "complete": self._error is None,
                "writer_dropped": self._dropped,
                "writer_error": self._error,
            },
        }
        self._config.update(
            {
                "ended_at_ns": time.time_ns(),
                "runtime": {
                    "termination_reason": termination_reason,
                    "actions": int(actions),
                    "chunks": int(chunks),
                    "event_counts": self._counts,
                    "artifacts": artifacts,
                },
                "session_result": {
                    "result": "unlabeled",
                    "failure_type": None,
                    "notes": "",
                    "labeled_at_ns": time.time_ns(),
                    "source": "non_interactive_runtime",
                },
            }
        )
        _atomic_json(self.config_path, self._config)


# Compatibility alias for pre-refactor imports.
OpenVLAInferenceSessionLogger = InferenceSessionLogger
