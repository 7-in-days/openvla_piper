#!/usr/bin/env python3
"""Verify this runtime against the current piper_ws and logging contracts."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from openvla_pipeline.cli import Option, parse_options
from openvla_pipeline.inference_logger import InferenceSessionLogger


JOINT_KEYS = tuple(f"joint_{index}.pos" for index in range(1, 7)) + ("gripper.pos",)
COMMAND_NAMES = tuple(f"joint{index}" for index in range(1, 7)) + ("gripper",)


def _evaluate(node: ast.AST, values: dict[str, object]) -> object:
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate(item, values) for item in node.elts)
    if isinstance(node, ast.List):
        return [_evaluate(item, values) for item in node.elts]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _evaluate(node.left, values) + _evaluate(node.right, values)  # type: ignore[operator]
    return ast.literal_eval(node)


def _values(path: Path, class_name: str | None = None) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = tree.body
    if class_name is not None:
        classes = [
            node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        if len(classes) != 1:
            raise AssertionError(f"class {class_name!r} not found exactly once in {path}")
        body = classes[0].body
    result: dict[str, object] = {}
    for node in body:
        name: str | None = None
        value_node: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                name, value_node = target.id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value_node = node.target.id, node.value
        if name is None or value_node is None:
            continue
        try:
            result[name] = _evaluate(value_node, result)
        except (ValueError, TypeError, KeyError):
            continue
    return result


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise AssertionError(f"required file is missing: {path}")
    return path


def _assert_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise AssertionError(f"{label} mismatch: actual={actual!r}, expected={expected!r}")


def _logging_probe(log_root: Path) -> None:
    session_id = f"contract_probe_{os.getpid()}"
    logger = InferenceSessionLogger(
        log_root,
        session_id,
        {"client": {"mode": "contract-probe"}},
    )
    closed = False
    try:
        logger.append_chunk(
            chunk_id=0,
            request_stamp_ns=100,
            observation_packet={
                "obs_third_person_stamp_ns": 10,
                "obs_wrist_stamp_ns": 11,
                "obs_qmeas_stamp_ns": 12,
            },
            instruction="contract probe",
            publish_stamp_ns=200,
        )
        logger.append_observability({"event": "contract_probe"})
        logger.close("normal", actions=1, chunks=1)
        closed = True
        config = json.loads(logger.config_path.read_text(encoding="utf-8"))
        core_rows = logger.log_path.read_text(encoding="utf-8").splitlines()
        events = logger.observability_path.read_text(encoding="utf-8").splitlines()
        _assert_equal("logger schema", config["schema_version"], 3)
        _assert_equal("core log rows", len(core_rows), 1)
        if len(events) < 3:
            raise AssertionError("observability log did not flush all lifecycle events")
    finally:
        if not closed:
            logger.close("error", actions=0, chunks=0)
        shutil.rmtree(logger.session_dir)


def main() -> None:
    args, _ = parse_options(
        None,
        (
            Option("piper_ws", converter=Path, default=Path("/home/pc/piper_ws")),
            Option("bridge_repo", converter=Path, default=None),
            Option("log_root", converter=Path, default=PROJECT_ROOT / "inference_logs"),
        ),
        description="Verify piper_ws topics, units, QoS, and inference logging.",
    )

    bridge_repo = args.bridge_repo
    if bridge_repo is None:
        pointer = PROJECT_ROOT / ".piper-repo"
        bridge_repo = (
            Path(pointer.read_text(encoding="utf-8").strip())
            if pointer.is_file()
            else Path.home() / "vla_pipeline"
        )

    executor = _values(
        _require_file(args.piper_ws / "src/executor_pkg/executor_pkg/piper_policy_executor.py")
    )
    recorder = _values(
        _require_file(args.piper_ws / "src/rollout_record_pkg/rollout_record_pkg/config.py")
    )
    recorder_contract_path = _require_file(
        args.piper_ws / "src/rollout_record_pkg/rollout_record_pkg/topic_contract.py"
    )
    bridge = _values(
        _require_file(bridge_repo / "piper_bridge/config_piper_bridge.py"),
        "PiperBridgeRobotConfig",
    )
    local_chunks = _values(_require_file(PROJECT_ROOT / "openvla_pipeline/chunk_topics.py"))

    _assert_equal("action topic (executor)", executor["ACTION_TOPIC"], "/piper/inference/output")
    _assert_equal("action names (executor)", executor["EXPECTED_ACTION_NAMES"], COMMAND_NAMES)
    _assert_equal("action topic (recorder)", recorder["ACTION_TOPIC"], executor["ACTION_TOPIC"])
    _assert_equal("original chunk topic", recorder["CHUNK_TOPIC"], local_chunks["CHUNK_TOPIC"])
    _assert_equal(
        "aggregated chunk topic",
        recorder["AGGREGATED_CHUNK_TOPIC"],
        local_chunks["AGGREGATED_CHUNK_TOPIC"],
    )
    _assert_equal(
        "chunk type (piper_ws)",
        recorder["CHUNK_MESSAGE_TYPE"],
        "trajectory_msgs/msg/JointTrajectory",
    )
    _assert_equal(
        "chunk type (rosbridge)",
        local_chunks["MESSAGE_TYPE"],
        "trajectory_msgs/JointTrajectory",
    )
    _assert_equal("SyncedFrame topic", bridge["frame_topic"], "/piper/synced/frame")
    _assert_equal("output topic (bridge)", bridge["output_topic"], executor["ACTION_TOPIC"])
    _assert_equal("bridge joint keys", bridge["joint_keys"], JOINT_KEYS)
    _assert_equal("bridge command names", bridge["command_out_names"], COMMAND_NAMES)
    _assert_equal("control rate", float(bridge["fps"]), 20.0)
    camera_names = tuple(camera[0] for camera in bridge["ros_cameras"])  # type: ignore[union-attr]
    if set(camera_names) != {"wrist", "third_person"}:
        raise AssertionError(f"camera contract mismatch: {camera_names!r}")
    contract_text = recorder_contract_path.read_text(encoding="utf-8")
    for token in ("ReliabilityPolicy.RELIABLE", "DurabilityPolicy.VOLATILE"):
        if token not in contract_text:
            raise AssertionError(f"recorder QoS token missing: {token}")

    _logging_probe(args.log_root.resolve())
    rollout_root = args.piper_ws / "bag/rollout/sessions"
    writable_parent = rollout_root if rollout_root.exists() else args.piper_ws

    print("piper_ws_contract=True")
    print("observation=/piper/synced/frame:piper_msgs/SyncedFrame")
    print("action=/piper/inference/output:sensor_msgs/JointState:7D")
    print("chunks=original+aggregated:trajectory_msgs/JointTrajectory")
    print("units=arm_radian,gripper_meter")
    print("names=" + ",".join(COMMAND_NAMES))
    print("cameras=" + ",".join(camera_names))
    print("control_hz=20")
    print("chunk_qos=reliable,volatile")
    print("openvla_log_write_flush_cleanup=True")
    print(f"piper_rollout_root_writable={os.access(writable_parent, os.W_OK)}")
    print(f"bridge_repo={bridge_repo.resolve()}")


if __name__ == "__main__":
    main()
