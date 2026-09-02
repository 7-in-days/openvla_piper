"""Run the inference-side OpenVLA pipeline with task-oriented subcommands."""

from __future__ import annotations

import runpy
import sys


COMMANDS = {
    "show-config": ("openvla_pipeline.config", ()),
    "model-server": ("openvla_pipeline.model_server", ()),
    "dry-run": ("openvla_pipeline.ros_node", ("--mode", "dry-run")),
    "run-robot": ("openvla_pipeline.ros_node", ("--mode", "live")),
}

COMPATIBILITY_COMMANDS = {
    "config": ("openvla_pipeline.config", ()),
    "serve": ("openvla_pipeline.model_server", ()),
    "deploy": ("openvla_pipeline.ros_node", ()),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        print(
            "usage: python -m openvla_pipeline "
            "{show-config|model-server|dry-run|run-robot} [options]"
        )
        print("compatibility aliases: config, serve, deploy")
        return
    command = sys.argv[1]
    command_spec = COMMANDS.get(command) or COMPATIBILITY_COMMANDS.get(command)
    if command_spec is None:
        choices = ", ".join(COMMANDS)
        raise SystemExit(f"unknown command {command!r}; expected one of: {choices}")
    module, fixed_arguments = command_spec
    sys.argv = [
        f"python -m openvla_pipeline {command}",
        *sys.argv[2:],
        *fixed_arguments,
    ]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
